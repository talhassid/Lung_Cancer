import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
import dicom  # for reading dicom files
from glob import glob
import pandas as pd  # just to load in the labels data and quickly reference it
import re
from skimage.measure import label, regionprops
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras import backend as K

def save_data(working_path: str, set_name: str,output_path)-> None:
    """
    Saving the data in the working path in npy file.

    :param working_path: The data path
    :param set_name: test/train
    :param output_path: The output path
    :return: None
    """
    patients = os.listdir(working_path)
    for num, patient in enumerate(patients):
        if os.path.isfile("/media/talhassid/Elements/haimTal/Unet/{}_images_{}.npy)".format(set_name,patient)):
            continue
        path = working_path + patient
        images = [dicom.read_file(path + '/' + s).pixel_array for s in os.listdir(path)]
        np.save("{}{}_images_{}.npy".format(output_path,set_name,patient),images)

def load_data()-> None:
    """
    Loading the data calling to save data.

    :return: None
    """
    output_path = '/media/talhassid/Elements/haimTal/Unet/'
    working_path_test = '/media/talhassid/Elements/haimTal/stage2/stage2/'
    save_data(working_path_test,"test",output_path)
    working_path_train = '/media/talhassid/Elements/haimTal/train/'
    save_data(working_path_train,"train",output_path)

def create_lungmask(file_list)-> None:
    """
    Creating a mask on the scan that will produce a "clean" lung image.

    :param file_list: List of the Lungs paths
    :return: None
    """

    for img_file in file_list:
        if os.path.isfile(re.sub(r"(.*)images(.*)",r"\1lungmask\2",img_file)):
            continue
        # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
        imgs_to_process = np.load(img_file).astype(np.float64)
        print ("on image", img_file)
        for i in range(len(imgs_to_process)):
            img = imgs_to_process[i]
            #Standardize the pixel values
            mean = np.mean(img)
            std = np.std(img)
            img = img-mean
            img = img/std
            # Find the average pixel value near the lungs
            # to renormalize washed out images
            middle = img[100:400,100:400]
            mean = np.mean(middle)
            max = np.max(img)
            min = np.min(img)
            # To improve threshold finding, I'm moving the
            # underflow and overflow on the pixel spectrum
            img[img==max]=mean
            img[img==min]=mean
            #
            # Using Kmeans to separate foreground (radio-opaque tissue)
            # and background (radio transparent tissue ie lungs)
            # Doing this only on the center of the image to avoid
            # the non-tissue parts of the image as much as possible
            #
            kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
            centers = sorted(kmeans.cluster_centers_.flatten())
            threshold = np.mean(centers)
            thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
            #
            # I found an initial erosion helpful for removing graininess from some of the regions
            # and then large dialation is used to make the lung region
            # engulf the vessels and incursions into the lung cavity by
            # radio opaque tissue
            #
            eroded = morphology.erosion(thresh_img,np.ones([4,4]))
            dilation = morphology.dilation(eroded,np.ones([10,10]))
            #
            #  Label each region and obtain the region properties
            #  The background region is removed by removing regions
            #  with a bbox that is to large in either dimension
            #  Also, the lungs are generally far away from the top
            #  and bottom of the image, so any regions that are too
            #  close to the top and bottom are removed
            #  This does not produce a perfect segmentation of the lungs
            #  from the image, but it is surprisingly good considering its
            #  simplicity.
            #
            labels = measure.label(dilation)
            label_vals = np.unique(labels)
            regions = measure.regionprops(labels)
            good_labels = []
            for prop in regions:
                B = prop.bbox
                if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                    good_labels.append(prop.label)
            mask = np.ndarray([512,512],dtype=np.int8)
            mask[:] = 0
            #
            #  The mask here is the mask for the lungs--not the nodes
            #  After just the lungs are left, we do another large dilation
            #  in order to fill in and out the lung mask
            #
            for N in good_labels:
                mask = mask + np.where(labels==N,1,0)
            mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
            imgs_to_process[i] = mask
        np.save(img_file.replace("images","lungmask"),imgs_to_process)

def mask_the_images(working_path,set_name)-> None:
    """
    Here we're applying the masks and cropping and resizing the image.

    :param working_path: The path of the data.
    :return: None
    """

    file_list=glob(working_path+"lungmask_*.npy")
    count = 0
    out_images = []  #final set of images for all patients
    for fname in file_list:
        print ("working on group ", count)
        count = count + 1
        out_images_per_patient = []
        print ("working on file ", fname)
        imgs_to_process = np.load(fname.replace("lungmask","images")) # images of one patient
        try:
            masks = np.load(fname)
            for i in range(len(imgs_to_process)):
                mask = masks[i]
                img = imgs_to_process[i]
                new_size = [512,512]   # we're scaling back up to the original size of the image
                img= mask*img          # apply lung mask
                #
                # renormalizing the masked image (in the mask region)
                #
                new_mean = np.mean(img[mask>0])
                new_std = np.std(img[mask>0])
                #
                #  Pulling the background color up to the lower end
                #  of the pixel range for the lungs
                #
                old_min = np.min(img)       # background color
                img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
                img = img-new_mean
                img = img/new_std
                #make image bounding box  (min row, min col, max row, max col)
                labels = measure.label(mask)
                regions = measure.regionprops(labels)
                #
                # Finding the global min and max row over all regions
                #
                min_row = 512
                max_row = 0
                min_col = 512
                max_col = 0
                for prop in regions:
                    B = prop.bbox
                    if min_row > B[0]:
                        min_row = B[0]
                    if min_col > B[1]:
                        min_col = B[1]
                    if max_row < B[2]:
                        max_row = B[2]
                    if max_col < B[3]:
                        max_col = B[3]
                width = max_col-min_col
                height = max_row - min_row
                if width > height:
                    max_row=min_row+width
                else:
                    max_col = min_col+height
                #
                # cropping the image down to the bounding box for all regions
                # (there's probably an skimage command that can do this in one line)
                #
                img = img[min_row:max_row,min_col:max_col]
                mask =  mask[min_row:max_row,min_col:max_col]
                if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
                    pass
                else:
                    # moving range to -1 to 1 to accomodate the resize function
                    mean = np.mean(img)
                    img = img - mean
                    min = np.min(img)
                    max = np.max(img)
                    img = img/(max-min)
                    new_img = resize(img,[512,512], mode='constant')
                    out_images_per_patient.append(new_img)
        except ValueError as e:
            id = re.sub(r'.*_images_(.*)\.npy',r'\1',fname)
            print("patient {} did some troubles".format(id))
            print('exception msg: '+ str(e))
            continue
        id = re.sub(r'.*_images_(.*)\.npy',r'\1',fname)
        patient_images_and_id = (out_images_per_patient,id)
        out_images.append(patient_images_and_id)
        print ("Delete files: {} \n\t {} ".format(fname,re.sub("lungmask","images",fname)))
        os.remove(fname)
        os.remove(re.sub(r"(.*)lungmask(.*)",r"\1images\2",fname))

        if (count % 10 == 0):

            np.save(working_path+"{}Images{}.npy".format(set_name,count),out_images)

def _mask_the_images(working_path, set_name, output_path):
    """
    Here we're applying the masks and cropping and resizing the image


    :param working_path:
    :return:
    """

    file_list=glob(working_path+"lungmask_*.npy")
    for fname in file_list:
        out_images_per_patient = []
        print ("working on file ", fname)
        imgs_to_process = np.load(fname.replace("lungmask","images")) # images of one patient
        try:
            masks = np.load(fname)
            for i in range(len(imgs_to_process)):
                mask = masks[i]
                img = imgs_to_process[i]
                new_size = [512,512]   # we're scaling back up to the original size of the image
                img= mask*img          # apply lung mask
                #
                # renormalizing the masked image (in the mask region)
                #
                new_mean = np.mean(img[mask>0])
                new_std = np.std(img[mask>0])
                #
                #  Pulling the background color up to the lower end
                #  of the pixel range for the lungs
                #
                old_min = np.min(img)       # background color
                img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
                img = img-new_mean
                img = img/new_std
                #make image bounding box  (min row, min col, max row, max col)
                labels = measure.label(mask)
                regions = measure.regionprops(labels)
                #
                # Finding the global min and max row over all regions
                #
                min_row = 512
                max_row = 0
                min_col = 512
                max_col = 0
                for prop in regions:
                    B = prop.bbox
                    if min_row > B[0]:
                        min_row = B[0]
                    if min_col > B[1]:
                        min_col = B[1]
                    if max_row < B[2]:
                        max_row = B[2]
                    if max_col < B[3]:
                        max_col = B[3]
                width = max_col-min_col
                height = max_row - min_row
                if width > height:
                    max_row=min_row+width
                else:
                    max_col = min_col+height
                #
                # cropping the image down to the bounding box for all regions
                # (there's probably an skimage command that can do this in one line)
                #
                img = img[min_row:max_row,min_col:max_col]
                mask =  mask[min_row:max_row,min_col:max_col]
                if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
                    pass
                else:
                    # moving range to -1 to 1 to accomodate the resize function
                    mean = np.mean(img)
                    img = img - mean
                    min = np.min(img)
                    max = np.max(img)
                    img = img/(max-min)
                    new_img = resize(img,[512,512], mode='constant')
                    out_images_per_patient.append(new_img)
        except ValueError as e:
            id = re.sub(r'.*_images_(.*)\.npy',r'\1',fname)
            print("patient {} did some troubles".format(id))
            print('exception msg: '+ str(e))
            continue
        id = re.sub(r'.*_lungmask_(.*)\.npy',r'\1',fname)
        patient_images_and_id = (out_images_per_patient,id)
        np.save(output_path + "{}Images{}.npy".format(set_name,id),patient_images_and_id)
        print ("Delete files: {} \n\t {} ".format(fname,re.sub("lungmask","images",fname)))
        os.remove(fname)
        os.remove(re.sub(r"(.*)lungmask(.*)",r"\1images\2",fname))


K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.


def _dice_coef(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def _dice_coef_np(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def _dice_coef_loss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    return -_dice_coef(y_true, y_pred)

def get_unet():
    """
    Create the Unet layers.

    :return: Unet
    """
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=_dice_coef_loss, metrics=[_dice_coef])

    return model

def get_mask_from_unet(output_path,data,set_name)-> None:
    """
    Create masked lungs by prediction of the Unet.

    :param output_path: The output path of the masked lungs.
    :param data: The path of the data we want to get a prediction on.
    :param set_name: test/train
    :return: None
    """
    print('-'*30)
    print('Loading data...')
    print('-'*30)
    imgs_test_and_ids = np.load(data)

    print('-'*30)
    print('compiling model...')
    print('-'*30)
    model = get_unet()

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('/home/talhassid/PycharmProjects/Lung_Cancer/unet.hdf5')

    print('-'*30)
    print('Predicting masks on data...')
    print('-'*30)

    num_patients = len(imgs_test_and_ids)


    for i in range(num_patients):
        num_test = len(imgs_test_and_ids[i][0])
        imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
        imgs_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
        for j in range(num_test):
            imgs_test[j,0] = imgs_test_and_ids[i][0][j]
            imgs_mask_test[j] = model.predict(imgs_test, verbose=0)[0]
        np.save('{}{}_mask_predicted_{}.npy'.format(output_path,set_name,imgs_test_and_ids[i][1]), imgs_mask_test)
        # print ("Delete file: {} ".format(data))
        # os.remove(data)

def _get_mask_from_unet(output_path, data, set_name):
    """

    :param output_path:
    :param data:
    :param set_name:
    :return:
    """
    print('-'*30)
    print('Loading data...')
    print('-'*30)
    imgs_test_and_id = np.load(data)

    print('-'*30)
    print('compiling model...')
    print('-'*30)
    model = get_unet()

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('/home/talhassid/PycharmProjects/Lung_Cancer/unet.hdf5')

    print('-'*30)
    print('Predicting masks on data...')
    print('-'*30)


    images = imgs_test_and_id[0]
    id = imgs_test_and_id[1]

    num_test = len(images)
    imgs_test = np.ndarray([num_test,1,512,512],dtype=np.float32)

    for i in range(num_test):
        imgs_test[i,0] = images[i]

    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict(imgs_test[i:i+1], verbose=0)[0]
    np.save('{}{}_mask_predicted_{}.npy'.format(output_path,set_name,id), imgs_mask_test)


if __name__ == '__main__':
    working_path = '/media/talhassid/Elements/haimTal/Unet/'
    mask_the_images(working_path+"train_","train")
    #file_list = ["/media/talhassid/Elements/haimTal/Unet/test_lungmask_1cf841414a1eda13138f2493ecaf439c.npy"]

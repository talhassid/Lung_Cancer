import os
import re
import scipy

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import dicom  # for reading dicom files
import numpy as np
import tensorflow as tf
import os
from skimage.morphology import ball, disk, binary_erosion, binary_closing
from skimage.measure import label,regionprops
from skimage.filters import roberts
from skimage import measure
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

###############################################process data#########################################################
IMG_PX_SIZE = 100 #to make the slices in same size.
SLICE_COUNT = 20  #numbers of slices in each chunk.


def load_scan(patient,data_dir):
    """
    Loaing the dicom scans of patient.
    
    :param patient: Patient id
    :param data_dir: The patients dir
    :return: The scans
    """

    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2])) # sorting the dicom by x image position
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    """
    Convert slices pixels to HU units.

    :param slices: Scans of patient.
    :return: HU pixel array of the scans
    """
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan,img_px_size=50,hm_slices=20):
    """
    Resampling image to be in the following shape: img_px_size * img_px_size * hm_slices

    :param image: HU pixel array
    :param scan: Original scan
    :param img_px_size: The size we want the image to be, default = 100
    :param hm_slices: The amount of image we use, default = 20
    :return: resampled image, the spacing
    """
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    new_shape = np.array([hm_slices,img_px_size,img_px_size])
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def _plot_3d(image, threshold=-300):
    """
    Ploting 3D

    :param image:
    :param threshold:
    :return: None
    """

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    verts, faces = measure.marching_cubes_classic(p, threshold)
    #verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def _normalize(image):
    """

    :param image:
    :return:
    """
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

PIXEL_MEAN = 0.25

def _zero_center(image):
    """

    :param image:
    :return:
    """
    image = image - PIXEL_MEAN
    return image

def _process_data_new(patient, data_dir, img_px_size, hm_slices):
    """

    :param patient:
    :param data_dir:
    :param img_px_size:
    :param hm_slices:
    :return:
    """
    image = np.load(data_dir+patient)
    image = image[:,0,:,:]
    # plot_3d(image, 0)
    new_shape = np.array([hm_slices,img_px_size,img_px_size])
    shape = image.shape
    real_resize_factor = new_shape / shape
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image

def process_data(patient,data_dir,img_px_size,hm_slices):
    """
    Process patient scans.

    :param patient: Id of patient
    :param data_dir: Dir of patients
    :param img_px_size: The size we want the image to be, default = 100
    :param hm_slices: The amount of image we use, default = 20
    :return: processed 3D image
    """
    img_data = load_scan(patient,data_dir,img_px_size=img_px_size, hm_slices=hm_slices)
    img_pixels = get_pixels_hu(img_data)
    img_pix_resampled, spacing = resample(img_pixels, img_data, img_px_size=img_px_size, hm_slices=hm_slices)
    #plot_3d(img_pix_resampled, 400)
    # img_pix_resampled = normalize(img_pix_resampled)
    # img_pix_resampled = zero_center(img_pix_resampled)
    return  img_pix_resampled

def load_and_process_data(patients,labels_df,data_dir,file_name,train_flag)-> None:
    """
    Loading the images and preprocees them.

    :param patients: List of patients id's
    :param labels_df: Labels file
    :param data_dir: Parent dir
    :param file_name: The name of the ouput file
    :param train_flag: Flag - if the data is from train set
    :return: None
    """
    print ("starting process data...\n")
    much_data = []
    #just to know where we are, each 100 patient we will print out
    for num, patient in enumerate(patients):
        id = re.sub(r'.*_mask_predicted_(.*)\.npy',r'\1',patient)
        if num%10==0:
            print(num)
        try:
            img_data = _process_data_new(patient, data_dir, img_px_size=IMG_PX_SIZE, hm_slices=SLICE_COUNT) #added new
            if (train_flag):
                label = labels_df.get_value(id, 'cancer') #the value for the cancer column
                #left column nocancer,right column cancer
                if label == 1: label=np.array([0,1])
                elif label == 0: label=np.array([1,0])
                much_data.append([img_data,label,id])
            else:
               much_data.append([img_data,id])
        except KeyError as e:
            print('This is unlabeled data!')

    np.save('/media/talhassid/Elements/haimTal/run/{}-{}-{}-{}.npy'
            .format(file_name,IMG_PX_SIZE,IMG_PX_SIZE,SLICE_COUNT), much_data)

###############################################building the net######################################################
def _conv3d(x, W):
    """

    :param x:
    :param W:
    :return:
    """
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def _maxpool3d(x):
    """

    :param x: Data
    :return: maxpool of tensorflow module
    """
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x, keep_rate=0.8, n_classes=2):
    """
    Building our cnn layers

    :param x: Tensor placeholder
    :param keep_rate: default_value=0.8
    :param n_classes: number of classes, default value=2
    :return: output Layer
    """
    #                # 3 x 3 x 3 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       3 x 3 x 3 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
                'W_conv3':tf.Variable(tf.random_normal([3,3,3,64,128])),
               'W_conv4':tf.Variable(tf.random_normal([3,3,3,128,256])),
               'W_conv5':tf.Variable(tf.random_normal([3,3,3,256,512])),
               'W_conv6':tf.Variable(tf.random_normal([3,3,3,512,1024])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([4096,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_conv3':tf.Variable(tf.random_normal([128])),
              'b_conv4':tf.Variable(tf.random_normal([256])),
              'b_conv5':tf.Variable(tf.random_normal([512])),
              'b_conv6':tf.Variable(tf.random_normal([1024])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_PX_SIZE, IMG_PX_SIZE, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(_conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = _maxpool3d(conv1)

    conv2 = tf.nn.relu(_conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = _maxpool3d(conv2)

    conv3 = tf.nn.relu(_conv3d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = _maxpool3d(conv3)

    conv4 = tf.nn.relu(_conv3d(conv3, weights['W_conv4']) + biases['b_conv4'])
    conv4 = _maxpool3d(conv4)

    conv5 = tf.nn.relu(_conv3d(conv4, weights['W_conv5']) + biases['b_conv5'])
    conv5 = _maxpool3d(conv5)

    conv6 = tf.nn.relu(_conv3d(conv5, weights['W_conv6']) + biases['b_conv6'])
    conv6 = _maxpool3d(conv6)

    fc = tf.reshape(conv6,[-1, 4096])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)


    output = tf.matmul(fc, weights['out'],name="op_to_restore")+biases['out'] #add name="op_to_restore"

    return output

###############################################train the net##########################################################

def train_neural_network(epochs_count=30, validation_count=100)-> None:
    """
    Train the net with the train set and keep trained net in output file.

    :param epochs_count: default value=30
    :param validation_count: default value=100
    :return: None
    """
# loading data
    print ("loading process data...\n")
    much_data = np.load('/media/talhassid/Elements/haimTal/run/train_masks-100-100-20.npy')
    print ("loading process data finished, starting the training...\n")
    train_data = much_data[:-validation_count] #2 for sampleimages and 100 for stage1
    validation_data = much_data[-validation_count:]
# the network
    x = tf.placeholder('float',name="input_tensor") # will consist a tensor of floating point numbers.
    y = tf.placeholder('float',name="target_tensor") # the target output classes will consist a tensor.
    output_layer = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
# training
    hm_epochs = epochs_count

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with open('/media/talhassid/Elements/haimTal/run/accuracy_results.txt', 'w') as a:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for data in train_data:
                    try:
                        X = data[0]
                        Y = data[1]
                        _, c= sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                        epoch_loss += c
                    except Exception as e:
                        pass
                print('Epoch', epoch+1, 'completed , loss:',epoch_loss)
                # find predictions on val set
                correct = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

            print('Done. Finishing accuracy:')
            print('Validation Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            print ("Training complete!")
            # Save the variables to disk.
            save_path = saver.save(sess, "/media/talhassid/Elements/haimTal/run/model.ckpt")
            print("Model saved in file: %s" % save_path)

############################################################testing#######################################
        # with open('/home/talhassid/PycharmProjects/lung_cancer/sentex/prediction.txt', 'w') as f:
        #     for index in range(10):
        #         test_data = much_data[index]
        #         X = test_data[0]
        #         Y = test_data[1]
        #         feed_dict = {x:X,y:Y}
        #         prediction=tf.nn.softmax(output_layer)
        #         print ("p_id:",much_data[index][2], "prediction[no_cancer , cancer]:",
        #                sess.run(prediction,feed_dict=feed_dict),file=f)

def test()-> None:
    """
    Clarify test set, printing the results

    :return: None
    """
    sess=tf.Session()
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('/media/talhassid/Elements/haimTal/run/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/media/talhassid/Elements/haimTal/run/'))

    graph = tf.get_default_graph()

    #Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    x_restore = graph.get_tensor_by_name("input_tensor:0")
    y_restore = graph.get_tensor_by_name("target_tensor:0")
    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    much_data = np.load('/media/talhassid/Elements/haimTal/run/test_masks-100-100-20.npy')


    with open('/media/talhassid/Elements/haimTal/run/test_results.csv', 'w') as f:
        print ("id,cancer",file=f)
        for index,_ in enumerate(much_data):
            test_data = much_data[index]
            X = test_data[0]
            feed_dict = {x_restore:X}
            prediction=tf.nn.softmax(op_to_restore)
            pred = sess.run(prediction,feed_dict=feed_dict)
            print (much_data[index][1],",",pred)
            print (much_data[index][1],",",pred[0][1],file=f)
            #print (much_data[index][1],",",(prediction.eval(session=sess,feed_dict=feed_dict)[0][1]))

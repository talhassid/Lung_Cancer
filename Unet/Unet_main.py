from glob import glob

import dicom
import os
import numpy as np

from Unet.Unet_Commands import create_lungmask, _mask_the_images, save_data, \
    _get_mask_from_unet, createFeatureDataset, classifyData

if __name__ == '__main__':


    """
    Loading the data
    """
    print("#"*30)
    print("Loading the data")
    print("#"*30)
    output_path = '/media/talhassid/Elements/haimTal/Unet/'

    working_path_test = '/media/talhassid/Elements/haimTal/stage2/stage2/'
    # save_data(working_path_test,"test",output_path)

    """
    preprocessing the data
    """
    print("#"*30)
    print("process the data")
    print("#"*30)
    working_path = '/media/talhassid/Elements/haimTal/Unet/'

    file_list_test=glob(working_path+"test_images_*.npy")
    #create_lungmask(file_list_test)
    # mask_the_images2(working_path+"test_","test",working_path+"/processed_data/")


    print("#"*30)
    print("getting mask from unet test")
    print("#"*30)
    net_input_test=glob(working_path+"/processed_data/testImages*.npy")
    import re


    for fname in net_input_test:
        count = re.sub(r'.*Images(.*)\.npy',r'\1',fname)
        imgs_test_and_id = np.load(working_path+"/processed_data/testImages{}.npy".format(count))
        id = imgs_test_and_id[1]
        tmp_file="/media/talhassid/Elements/haimTal/Unet/masks/{}_mask_predicted_{}.npy".format("test",id)
        if os.path.isfile(tmp_file):
            print("I've continued")
            continue
        _get_mask_from_unet(working_path + "/masks/", working_path + "/processed_data/testImages{}.npy".format(count), "test")


    """
    Loading the data
    """
    print("#"*30)
    print("Loading the data")
    print("#"*30)

    working_path_train = '/media/talhassid/Elements/haimTal/train/'
    save_data(working_path_train,"train",output_path)

    file_list_train=glob(working_path+"train_images_*.npy")
    create_lungmask(file_list_train)
    _mask_the_images(working_path + "train_", "train", working_path + "/processed_data")

    """
    getting mask from unet
    """
    print("#"*30)
    print("getting mask from unet train")
    print("#"*30)
    net_input_train=glob(working_path+"/processed_data/trainImages*.npy")
    import re
    for fname in net_input_train:
        count = re.sub(r'.*Images(.*)\.npy',r'\1',fname)
        imgs_test_and_id = np.load(working_path+"/processed_data/testImages{}.npy".format(count))
        id = imgs_test_and_id[1]
        tmp_file="/media/talhassid/Elements/haimTal/Unet/masks/{}_mask_predicted_{}.npy".format("train",id)
        if os.path.isfile(tmp_file):
            print("I've continued")
            continue
        _get_mask_from_unet(working_path + "/masks/", working_path + "/processed_data/trainImages{}.npy".format(count), "train")



    print("#"*30)
    print("classify by known nets")
    print("#"*30)
    nodfiles = '/media/talhassid/Elements/haimTal/Unet/masks/*_masks_predicted_*.npy'
    createFeatureDataset(nodfiles)
    classifyData()

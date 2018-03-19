import os

from Our_CNN.CNN_Commands import train_neural_network, test, load_and_process_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os  # for doing directory operations
import pandas as pd  # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
if __name__ == '__main__':

    EPOCHS_COUNT = 30
    VALIDATION_COUNT = 20 #hm patients will be in validation

    labels_df = pd.read_csv('/media/talhassid/Elements/haimTal/stage2_labels.csv', index_col=0)

    data_dir_train = '/media/talhassid/Elements/haimTal/run/train/'
    patients_train = os.listdir(data_dir_train)
    load_and_process_data(patients_train,labels_df,data_dir_train,"train_masks",train_flag=True)


    data_dir_test = '/media/talhassid/Elements/haimTal/run/test/'
    patients_test = os.listdir(data_dir_test)
    load_and_process_data(patients_test,labels_df,data_dir_test,"test_masks",train_flag=False)

    train_neural_network(epochs_count=EPOCHS_COUNT,validation_count=VALIDATION_COUNT)
    test()

    print ("finish")



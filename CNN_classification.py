import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image 
from pathlib import Path
from shutil import copy

import random
import time
import csv
from tensorflow.keras.utils import  plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPooling2D,Activation, Dropout, Flatten
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras import regularizers

## Fix the random seed
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)




class Image_data:
    ''' reads, saves, and load dataset files '''
    def __init__(self):
        self.size = 128
        self.img_array=[]
        self.filename_array=[]
        self.train_set = None
        self.val_set = None
        self.test_set = None
    def read_images(self, path, valid_type = [".JPEG", ".jpg"]):
        ## read image files, resize and convert them to numpy arrays
        tmp_arr = []
        filename = []
        path = Path(path)
        for image_name in path.absolute().iterdir():                                                      
            if image_name.suffix in valid_type:
                current_file = path / image_name 
                tmp_arr.append(np.asarray(Image.open(current_file).resize((self.size, self.size))))
                filename.append(current_file.name)
                if tmp_arr[-1].shape == (self.size, self.size): ## find grayscale images, and convert them to 3D
                    tmp_arr[-1] = np.repeat(tmp_arr[-1][:, :, np.newaxis], 3, axis=2)
                if tmp_arr[-1].shape != (self.size, self.size, 3): ## some had 4 dimensions, image index 19876 imagenet_128
                    # print(image_name, tmp_arr[-1].shape)
                    tmp_arr.pop()
                    filename.pop()   
        self.img_array = np.array(tmp_arr)
        self.filename_array = np.array(filename)
        return self
    def load(self, path = Path(), category_train = "White/Male", category_test ="Black/Female"):
        ## load data(train,val,test) .npz files if available, else create and save them
        train_face_dir = Path()/'fairface-img-margin025-trainval'/'mixed_features'/'train'/ Path(category_train)
        train_obj_dir = Path()/'imagenet_128' # Please download ImageNet files directly from http://image-net.org/challenges/LSVRC/, we are not allowed to share them
        val_face_dir = Path()/'fairface-img-margin025-trainval'/'mixed_features'/'val'/ Path(category_train)
        val_obj_dir = Path()/'imagenet_128_val'# Please download ImageNet files directly from http://image-net.org/challenges/LSVRC/, we are not allowed to share them
        test_face_dir = Path()/'fairface-img-margin025-trainval'/'mixed_features'/'val'/Path(category_test)


        dir_dict = {'train': [train_face_dir,train_obj_dir], 'val':[val_face_dir,val_obj_dir], 'test':[test_face_dir]}
        sample_size_dict = {'train': [800, 10000], 'val': [200, 1000], 'test': [len(os.listdir(dir_dict['test'][0]))]} # Org : 800 train, 200 val, all for test
        #sample_size_dict = {'train': [800, 10000], 'val': [len(os.listdir(dir_dict['val'][0])), 1000], 'test': [len(os.listdir(dir_dict['test'][0]))]} # another experiment: 800 train, all val, all for test
        category_dict = {'train': category_train, 'val':category_train, 'test':category_test}
        data = {}
        for category in category_dict:  
            fname = category_dict[category].replace('/','_')
            sample_size_name = category
            for l in range(len(sample_size_dict[category])): ## creat file name containing size info
                sample_size_name+= '_' + str(sample_size_dict[category][l])
            file_name_category = f'{fname}_{sample_size_name}.npz'
            file_name = os.path.join(file_name_category)
            if os.path.exists(file_name): ## data file available, load it
                print(f'Loading saved files: {category}')
                data[category] = np.load(file_name_category)
            else: ## data file not available, generate and save it
                print(f'Reading files: {category}')
                self.read_images(dir_dict[category][0])
                face_array = self.img_array[:sample_size_dict[category][0]]
                face_filename = self.filename_array[:sample_size_dict[category][0]]
                
                ## save resized image files
                # mkdir_p(LOG_DIR/"images"/category)
                # for image_idx in range(len(face_array)):
                #     plt.imsave(LOG_DIR/"images"/category/face_filename[image_idx], face_array[image_idx]) 

                if len(dir_dict[category])> 1: #obj dir exists in the dictionary
                    self.read_images((dir_dict[category][1]))
                    obj_array = self.img_array[:sample_size_dict[category][1]]
                    obj_filename = self.filename_array[:sample_size_dict[category][1]]
                    data_array = np.concatenate((face_array,obj_array), axis=0)
                    data_filename = np.concatenate((face_filename,obj_filename), axis=0)
                    data_label = np.concatenate((np.zeros((len(face_array),1)),np.ones((len(obj_array),1))))
                else:
                    data_array = face_array ## true for test set category which doesn't have obj 
                    data_filename = face_filename
                    data_label = np.zeros((len(data_array),1))   
                np.savez(file_name_category, data=data_array, label=data_label, filename = data_filename)
                data[category] = np.load(file_name_category)

        self.train_set = data['train'] ## contains data, label and filename
        self.val_set = data['val']
        self.test_set = data['test']
        return self
        
class Classifier:
    ''' makes train, validation, test data ready
        defines classifiers
        trains and tests the model
    '''
    def __init__(self):
        self.input_shape = None
        self.model = None
        self.history = None

        self.train_data = None
        self.train_label = None
        self.train_filename = None
        self.val_data = None
        self.val_label = None
        self.val_filename = None
        self.test_data = None
        self.test_label = None
        self.test_filename = None
    def prep_data(self, category_train = "White/Male", category_test ="Black/Female"):
        ## prepare data for classification
        reader = Image_data()
        reader.load(category_train= category_train, category_test=category_test)
        self.train_data = reader.train_set['data']
        self.train_label = reader.train_set['label']
        self.train_filename = reader.train_set['filename']

        self.val_data = reader.val_set['data']
        self.val_label = reader.val_set['label']
        self.val_filename = reader.val_set['filename']

        self.test_data = reader.test_set['data']
        self.test_label = reader.test_set['label']
        self.test_filename = reader.test_set['filename']

        self.input_shape = (reader.size,reader.size,3)     

    def cnn_model(self):
        ## CNN model architecture
        model = Sequential()

        model.add(Conv2D(filters=128, kernel_size=(8,8), strides=(4,4), padding='same', input_shape=self.input_shape,data_format='channels_last', name='conv1'))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=128, kernel_size=(8,8), strides=(4,4), padding='same', name='conv2'))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=256, kernel_size=(8,8), strides=(4,4), padding='valid', name='conv3'))
        model.add(Activation('relu'))
        model.add(Flatten())  

        model.add(Dense(1, name='dense_out'))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    

        ## print CNN model diagram
        mkdir_p(LOG_DIR)
        plot_model(model, show_shapes=True, to_file='cnn_model_new.png')

        #print(model.summary())
        self.model = model
        return self      
    def get_intermadiate_output(self, data, layer_name):
        ## copying the model till the desired layer and transfer the weights for intermadiate output

        dict_layer = {v.name: i for i, v in enumerate(self.model.layers)} 
        layer_idx = dict_layer[layer_name] ## find index corresponding to the layer name

        new_model = Sequential()
        for layer in self.model.layers[0:layer_idx+1]:
            new_model.add(layer)

        for l_tg,l_sr in zip(new_model.layers,self.model.layers):
            wk0 = l_sr.get_weights()
            l_tg.set_weights(wk0)
            if l_tg.name == layer_name:
                break

        intermediate_output = np.empty((data.shape[0],1))    
        if data.shape[0] > 2000: # o.w. Out of Memory error
            process_in_batch = True
        else:
            process_in_batch = False    

        if process_in_batch == False:                          
            intermediate_output = new_model.predict(data)
        else:
            batch_size = 100
            for b in range(0, data.shape[0], batch_size):   #To Do: conditional for edge cases len//batch_size?
                intermediate_output[b:b+batch_size] = new_model.predict(data[b:b+batch_size])
        
        return intermediate_output
    def save_intermadiate_output(self, layer_name ='dense_out'):
        ## save the intermadiate output of the desired layer for all train, val and test sets
        print("Intermediate 2 output files...(processing)...")
        
        inter_train = self.get_intermadiate_output(self.train_data, layer_name)
        inter_val = self.get_intermadiate_output(self.val_data, layer_name)
        inter_test = self.get_intermadiate_output(self.test_data, layer_name)

        dict_inter = {'train': inter_train, 'val': inter_val, 'test': inter_test}
        dict_filename= {'train': self.train_filename, 'val': self.val_filename, 'test': self.test_filename}
        dict_label = {'train': self.train_label, 'val': self.val_label, 'test': self.test_label}

        for category in dict_inter: 
            curr_path = LOG_DIR/"inter_output"
            mkdir_p(curr_path)
            with open(curr_path/f"{category}_inter_output_{layer_name}.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['file name', f'{layer_name} layer output', 'label' ])
                for idx in range(len(dict_inter[category])):  
                    writer.writerow([dict_filename[category][idx], dict_inter[category][idx][0], dict_label[category][idx][0] ])
        
        print("Intermediate 2 output files saved")
        return None
    
    def evaluate_model(self, epochs = 35, batch_size = 100 ):
        ## evaluate model on train and validation set, plot the learning trend, and save the model
        print("Training the model ...(processing)...")
        history = self.model.fit(self.train_data,self.train_label,batch_size=batch_size,epochs=epochs,validation_data=(self.val_data, self.val_label), shuffle=True, verbose = 1)
        print("Training the model done.")
        self.history = history
        self.visualize_learning()
        self.model.save("saved_model")
        return self
    def load_saved_model(self, dir = 'saved_model'):
        # load the previously saved model
        loaded_model = tf.keras.models.load_model(dir)    
        self.model = loaded_model
        return self 
    def test_model(self):
        ## test the model performance on test set, save the output into CSV format
        acc_test = []
        loss_test = []
        keep_idx_failed = []
        mkdir_p(LOG_DIR/"failed"/"failed_images")
        with open(LOG_DIR/"failed"/"test_results.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['file name', 'loss', 'accuracy' ])
            for idx in range(len(self.test_data)):
                current_test_img = self.test_data[idx]
                test_4d = current_test_img[np.newaxis, :]
                result = self.model.evaluate(test_4d,self.test_label[idx],verbose=0)
                loss_test.append(result[0])
                acc_test.append(result[1])
                if result[1] < 1:
                    keep_idx_failed.append(idx)
                    plt.imsave(LOG_DIR/"failed"/"failed_images"/self.test_filename[idx], current_test_img)   
                writer.writerow([self.test_filename[idx], result[0], result[1]])

        acc_test = np.array(acc_test)
        print("average accuracy test", np.average(acc_test))
        loss_test = np.array(loss_test)

        # plot test trend
        plt.figure('test_loss')
        plt.plot(loss_test)
        plt.savefig(LOG_DIR/"test_loss.png")
        plt.close()

        title = 'test_accuracy' + str(np.average(acc_test))
        plt.figure(title)
        plt.plot(acc_test)
        plt.savefig(LOG_DIR/f"test_acc_{title}.png")
        plt.close()

        return  np.average(acc_test)
    def visualize_learning(self, title='learning' ):
        ## visulize the learning trend, train and validation
        path = LOG_DIR
        mkdir_p(path)
        # Plot training & validation accuracy values
        history = self.history
        plt.figure(title)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(path / f"{title}_acc.png")
        plt.clf()

        # Plot training & validation loss values
        plt.figure(title)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(path / f"{title}_loss.png")
        plt.clf()
        plt.close()


def mkdir_p(mypath):
    ## Creates a directory. equivalent to using mkdir -p on the command line

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

if __name__ == "__main__":

    begin = time.time()
    global current_time
    global LOG_DIR
    current_time = int(time.time())
    LOG_DIR =  Path()/'results_log'/f"{current_time}" #creating unique log directory for each pass

    ## experiment 1: train of white male, validate on white male, test on black female
    # run this block if you want to train a new model and save it
    category_train = "White/Male"
    category_test = "Black/Female"

    ''' train: use this block when the model is not saved before'''
    load_pretrained = False
    my_classifier = Classifier()
    my_classifier.prep_data(category_train, category_test)
    if load_pretrained == False:
        my_classifier.cnn_model()
        my_classifier.evaluate_model()
    else:    
        my_classifier.load_saved_model()
    my_classifier.save_intermadiate_output( layer_name ='dense_out')    
    
    ''' test: make load_pretrained true to call the saved model on test data without training '''
    ## run this block if you want to test your saved model
    load_pretrained = True
    ## results dir
    all_result_dir =  Path()/'results_log'
    ## experiment 2: loop through all ethnicity groups and genders and evaluate the model on them
    ## save the results in seperate folders
    with open(all_result_dir/"all_test_results.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Ethnicity', 'Gender', 'Accuracy' ])
            category_train = "White/Male"
            for ethnicity in ['Black', 'East Asian', 'Indian', 'Latino_Hispanic','Middle Eastern', 'Southeast Asian']:
                for gender in ['Female', 'Male']:


                    current_time = int(time.time())
                    LOG_DIR =  Path()/'results_log'/f"{current_time}"
                    
                    category_test = ethnicity +"/" + gender
                    print('test category: ', category_test)
                    my_classifier = Classifier()
                    my_classifier.prep_data(category_train, category_test)
                    if load_pretrained == False:
                        my_classifier.cnn_model()
                        my_classifier.evaluate_model()
                    else:    
                        my_classifier.load_saved_model()
                    acc_test = my_classifier.test_model()
                    writer.writerow([ethnicity, gender, acc_test])

                    my_classifier.save_intermadiate_output( layer_name ='dense_out')
                    #my_classifier.save_intermadiate_output( layer_name ='activation_3')
                    

    
    end = time.time()
    print("Done") 
    print(f"Total runtime of the program is {(end - begin)//60} minutes") 


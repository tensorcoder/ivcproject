import h5py
import numpy as np
from PIL import Image
import io
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import pickle
from itertools import zip_longest
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

batch_size = 32
input_height = 224
input_width = 224

new_model = tf.keras.models.load_model('ResNet18_model_BC')
new_model.summary()


def training_data(test_path):
    data_train=[]
    target_train=[]
    data_paths=[test_path] 
    for data_path in data_paths:
        categories=os.listdir(data_path)[0:2]
        labels=[i for i in range(len(categories))]
        label_dict=dict(zip(categories,labels)) #empty dictionary
        # print(label_dict)
        # print(categories)
        # print(labels)
        
        img_size=224
        for category in categories:
            path = os.path.join(data_path,category)
            for img in tqdm(os.listdir(path)):
                # try:
                    img_array = cv2.imread(os.path.join(path,img))
                    dim = (img_size, img_size)
                    img = cv2.resize(img_array, dim, interpolation = cv2.INTER_AREA)
                    new_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)          
                    new_img = new_img.astype('float32')
                    new_img = new_img/255.0

                    data_train.append(new_img)
                    target_train.append(label_dict[category])


    return np.array(data_train), np.array(target_train)

# test_images, test_labels=training_data()



data = []
testing_folder = 'data/A'
for perturb_type in os.listdir(testing_folder)[2:11]:
    # print(perturb_type)
    perturb_type_path = os.path.join(testing_folder, perturb_type)
    for perturb_level in os.listdir(perturb_type_path):
        # print(perturb_level)
        test_path = os.path.join(perturb_type_path, perturb_level)
        print(test_path)
        
        test_images, test_labels=training_data(test_path)
        # Evaluate the restored model
        loss, acc = new_model.evaluate(test_images, test_labels, verbose=1)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

        # print(new_model.predict(test_images).shape)

        # print("Test loss:", score[0])
        # print("Test accuracy:", score[1])
        data.append([test_path, acc])

# print(data)

with open('ResNet18_BC_results_on_A.pkl', 'wb') as f:
    pickle.dump(data, f)


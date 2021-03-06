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
import cv2
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.regularizers import l2
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from classification_models.tfkeras import Classifiers
from keras.models import Model
import pickle
from itertools import zip_longest, combinations
import tensorflow
KERAS_BACKEND=tensorflow
import keras
from sklearn.preprocessing import LabelEncoder 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import optuna
from keras.optimizers import RMSprop, Adam
from keras.backend import clear_session


def training_data(two_training_folders=['A', 'B']):
    data_train=[]
    target_train=[]
    data_paths=[f"data/{two_training_folders[0]}", f"data/{two_training_folders[1]}"] 
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
                data_train.append(new_img)
                target_train.append(label_dict[category])

    return np.array(data_train), np.array(target_train)

# data_train, target_train=training_data(two_training_folders=['C'])

def training_split(data_train, target_train, test_size):

    X_train, X_val, y_train, y_val = train_test_split(data_train, target_train, test_size=test_size, shuffle=True)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_train = np.reshape(X_train,(X_train.shape[0],224,224,3))
    X_val = np.reshape(X_val,(X_val.shape[0],224,224,3))
    # encode text category labels 
    # le = LabelEncoder() 
    # le.fit(y_train) 
    # y_train = le.transform(y_train) 
    # y_val = le.transform(y_val) 
    return X_train, X_val, y_train, y_val 

# X_train, X_val, y_train, y_val = training_split(data_train, target_train, test_size=0)

# print(X_train.shape)

# exit(0)

def res_net_18():
    ResNet18, preprocess_input = Classifiers.get('resnet18')

    res_net_18_model = ResNet18(include_top=False, weights='imagenet', input_shape=(224,224,3))

    output = res_net_18_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    
    for layer in res_net_18_model.layers:
        layer.trainable = False

    model = Sequential()

    model.add(res_net_18_model)

    model.add(Flatten())
    model.add(Dense(512,activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.summary() 
    return model




def objective(trial):
    clear_session()
    

    data_train, target_train = training_data(two_training_folders=['B', 'C'])
    
    val_split = trial.suggest_float('val_split', 0.2, 0.3)
    
    X_train, X_val, y_train, y_val  = training_split(data_train, target_train, test_size=val_split)
    model = res_net_18()

    lr = trial.suggest_float('lr', 1e-5, 1e-3)
    
    epochs = trial.suggest_int('epochs', 40, 60, 10)
    model.compile(optimizer = Adam(lr=lr), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=epochs, verbose=True, batch_size=5, validation_data=(X_val, y_val))
    
    score = model.evaluate(X_val, y_val, verbose=0)
    
    # model.save(f"ResNet18_model_{model_index}")
    # yield score[1]
    return score[1]

def plot_loss(history):
    plt.plot(history.history['loss'],'r',label='training loss')
    plt.plot(history.history['val_loss'],label='validation loss')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
# plot_loss(history)
    
def plot_acc(history):
    plt.plot(history.history['accuracy'],'r',label='training accuracy')
    plt.plot(history.history['val_accuracy'],label='validation accuracy')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
# plot_acc(history)



# def evaluate_model(X_val, y_val, model):
#     score = model.evaluate(X_val, y_val)
#     print("Test loss:", score[0])
#     print("Test accuracy:", score[1])
# # evaluate_model(X_val, y_val)



def train_model(two_training_folders, model_index, trial):
    # two_training_folders=['A','B']
    data_train, target_train=training_data(two_training_folders)
    X_train, X_val, y_train, y_val = training_split(data_train, target_train)
    model=res_net_18()
    history, accuracy, loss =compile_fit_model(model, model_index, X_train, y_train, X_val, y_val, trial)
    # loss_plot = plot_loss(history)
    # accuracy_plot = plot_acc(history)
    # model_evaluation = evaluate_model(X_val, y_val, model)



def main():
    # folders = ['A','B','C']
    # combs = combinations(folders, r=2)
    # training_folders = list(combs)
    
    

    # for two_training_folders in training_folders:

        
        #hyper parameter optimizer optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25, timeout=None)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
       
    
        # model_index = str(two_training_folders[0]+two_training_folders[1])
        # print(model_index)
        # train_model(two_training_folders, model_index)
    




main()



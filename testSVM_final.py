
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pylab as pl
from sklearn.metrics import confusion_matrix,accuracy_score
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from scipy.cluster.vq import vq
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
import optuna
from tqdm import tqdm
from keras.backend import clear_session
import pickle


def img_list(path):    
    return [os.path.join(path, f) for f in os.listdir(path)[0:2]]


def get_data(test_set):

    data_paths=[test_set] 
    image_paths = []
    image_classes = []
    for data_path in data_paths:
        categories=os.listdir(data_path)[0:2]
        labels=[i for i in range(len(categories))]
        label_dict=dict(zip(categories,labels)) #empty dictionary
        # print(label_dict)
        # print(categories)
        # print(labels)
        
        for category in categories:
            path = os.path.join(data_path, category)
            for img in tqdm(os.listdir(path)):
     
                image_paths.append(os.path.join(path, img))
                image_classes.append(label_dict[category])
        

    return image_paths, image_classes


def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
            x, y = kp.pt
            plt.imshow(cv2.circle(vis, (int(x), int(y)), 2, color))

def get_features(image_paths, k=200, iterations=1, stdslr='stdslr', voc='voc'):
    #orb features
    des_list=[]
    orb=cv2.ORB_create()
    for image_pat in image_paths:
        im=cv2.imread(image_pat)
        kp=orb.detect(im,None)
        keypoints,descriptor= orb.compute(im, kp)
        des_list.append((image_pat,descriptor))
    # descriptors=des_list[0][1]
    # for image_path,descriptor in des_list[0:]: #changed from [1:]
    #     descriptors=np.vstack((descriptors,descriptor))
    # descriptors_float=descriptors.astype(float)

    # print(descriptors_float)
    
    im_features=np.zeros((len(image_paths),k),"float32")
    for i in range(len(image_paths)):
        try:
            words,distance=vq(des_list[i][1],voc)
        except:
            try:
                print('second try catch')
                print(image_paths[i])
                words, distance = vq(des_list[i][1].astype(float), voc)
            except:
                print('continuing')
                continue
        finally:
            pass
        for w in words:
            im_features[i][w]+=1
    
    im_features=stdslr.transform(im_features)
    
    return im_features


def score_SVM_model(clf, X_test, y_test):
# load the model from disk
    result = clf.score(X_test, y_test)
    print(result)
    return result

def choose_test_folder(kfld):
    testing_folders = ['A', 'B', 'C']
    for folder in testing_folders:
        if folder not in kfld:
            return folder


def main():
    for kfld in ['AB', 'AC', 'BC']:
        filename = f"SVM_{kfld}_model.sav"
        thing1 = pickle.load(open(filename, 'rb'))
        clf = thing1[0]
        stdslr = thing1[1]
        voc = thing1[2]
        data = []
        testing_folder_name = choose_test_folder(kfld)
        testing_folder = f'data/{testing_folder_name}'
        for perturb_type in os.listdir(testing_folder)[2:11]:
            # print(perturb_type)
            perturb_type_path = os.path.join(testing_folder, perturb_type)
            for perturb_level in os.listdir(perturb_type_path):
                # print(perturb_level)
                test_path = os.path.join(perturb_type_path, perturb_level)

                # #uncomment line below to manually test a single folder
                # test_path = "data/C/gaussian_pixel_noise/18"

                print(test_path)
                image_paths, image_classes = get_data(test_path)
                # Evaluate the restored model
                X_test = get_features(image_paths, k=200, iterations=1, stdslr=stdslr, voc=voc)
                result = score_SVM_model(clf, X_test, image_classes)
                # print(new_model.predict(test_images).shape)

                # print("Test loss:", score[0])
                # print("Test accuracy:", score[1])
                data.append([test_path, result])

                # exit(1)
        with open(f"SVM_{kfld}_results_on_{testing_folder_name}.pkl", 'wb') as f:
            pickle.dump(data, f)
   
    

main()




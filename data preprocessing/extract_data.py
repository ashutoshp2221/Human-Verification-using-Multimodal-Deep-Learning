'''

This file contains utility functino that converts the face images and audio spectrograms into numpy arrays and returns them as features and labels.

'''



import numpy as np
import tensorflow as tf
import os
import cv2
from tqdm import tqdm

#converts your entire dataset of images into a list containing pixel values of the images as well as the class that image belongs to.

#tqdm shows your progress
def create_data(data_dir,categories,data,categorical_index,IMG_HEIGHT,IMG_WIDTH):
    
    for category in categories:
        path = os.path.join(data_dir+category)
        class_num = categories.index(category)
        categorical_index.append([category,class_num])
        
        for img in tqdm(os.listdir(path)):
            try:
                img_array_bgr = cv2.imread(os.path.join(path,img))
                img_array = cv2.cvtColor(img_array_bgr , cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array , (IMG_HEIGHT,IMG_WIDTH))
                data.append([new_array,class_num])
            except Exception as e:
                pass
                
    return data,categorical_index

# creating X and y from data list

def extract_features_and_labels(data):
    X = []
    y = []
    for features, labels in data:
        X.append(features)
        y.append(labels)

    # converting into numpy arrays
    X = np.array(X).reshape(-1,IMAGE_HEIGHT,IMAGE_WIDTH,3)
    y = np.array(y)

    return X,y

def randomize_zip(img_a,aud_a,img_b,aud_b,labels):

    temp = list(zip(img_a,aud_a,img_b,aud_b,labels)) 
    random.shuffle(temp) 
    img_a,aud_a,img_b,aud_b,labels = zip(*temp)

    return img_a,aud_a,img_b,aud_b,labels
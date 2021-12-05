#import library
import warnings
import matplotlib.pyplot as plt
import cv2
from keras.models import Model
import os
import glob
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,Dropout,Activation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from numpy import asarray
from numpy import clip
from PIL import Image

####read data and label

#readdddd
img_dir =""
data_path = os.path.join(img_dir,'*')

####files
files = glob.glob(data_path)
files=list(files)



data1 = []

####fffff
f=[]
label=[]

for i in range(len(files)):
    for j in glob.glob(files[i]+"/*.*"):
        
        label.append(i)

for f1 in glob.glob(data_path+"/*.*"):
    f.append(f1)
    img=cv2.imread(f1)
    img=cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
    #img=img/255
    data1.append(img)

print(len(data1))

print(len(label))

###split data1
data1=data1[0:10]
label=label[0:10]
d=[]
for i in range(len(data1)):
    d.append(data1[i]/255)

from sklearn import preprocessing



label_encoder = preprocessing.LabelEncoder()
label=label_encoder.fit_transform(label)



#### model





#make model
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Conv2D(filters=4096, kernel_size=(7,7), padding="same", activation="relu"))

model.add(Conv2D(filters=4096, kernel_size=(1,1), padding="same", activation="relu"))
model.add(Conv2D(filters=2622, kernel_size=(1,1), padding="same", activation="relu"))
model.add(Flatten())
model.add(Activation('softmax'))



##load model
from keras.models import model_from_json
model.load_weights('C:/Users/Downloads/vgg_weights.h5')



#build model and feature
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

feature=vgg_face.predict(np.array(d))


#svm model

####split
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size = 0.3)






#split to test and train


#make svm model
from sklearn import svm
svclassifier =svm.SVC(decision_function_shape='ovo')
svclassifier.fit(X_train,y_train)


#y_pred = svclassifier.predict(data1_)


#evaluations

print('score', svclassifier.score(X_test,y_test))



import os
import cv2
import numpy as np
from tensorflow import keras 
from keras.constraints import maxnorm
from keras.models import Sequential
import sklearn.model_selection as ms
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten,Conv2DTranspose,concatenate
from keras.utils import np_utils 
import matplotlib.pyplot as plt
import pandas as pd


xmls=[]
imgs=[]
YES = "/BrainTumor/YES"
NO =  '/BrainTumor/NO'

image_size = (32, 32) # اندازه تصاویر خروجی
num_images = len(os.listdir(YES)) 

YESimages = np.zeros((num_images,) + image_size + (2,), dtype=np.float32) # ایجاد لیست برای ذخیره تصاویر

# خواندن تصاویر و ذخیره در لیست images
for i, image_name in enumerate(os.listdir(YES)):
    image_path = os.path.join(YES, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # خواندن تصویر با OpenCV و تبدیل به سیاه و سفید
    image_resized = cv2.resize(image, image_size) # تغییر اندازه تصویر
    YESimages[i,:,:,0] = image_resized.astype(np.float32) # ذخیره کانال اول (سیاه و سفید)
    YESimages[i,:,:,1] = image_resized.astype(np.float32) # ذخیره کانال دوم (سیاه و سفید)

print(YESimages.shape) # شکل لیست تصاویر

image_size = (32, 32) # اندازه تصاویر خروجی
num_images = len(os.listdir(NO)) 

NOimages = np.zeros((num_images,) + image_size + (2,), dtype=np.float32) # ایجاد لیست برای ذخیره تصاویر

# خواندن تصاویر و ذخیره در لیست images
for i, image_name in enumerate(os.listdir(NO)):
    image_path = os.path.join(NO, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # خواندن تصویر با OpenCV و تبدیل به سیاه و سفید
    image_resized = cv2.resize(image, image_size) # تغییر اندازه تصویر
    NOimages[i,:,:,0] = image_resized.astype(np.float32) 
    NOimages[i,:,:,1] = image_resized.astype(np.float32) 

print(NOimages.shape) # شکل لیست تصاویر

datanum=NOimages.shape[0]+YESimages.shape[0]
datanum

YESimagesY = np.ones((YESimages.shape[0], 1))
NOimagesY = np.zeros((NOimages.shape[0], 1))

images = np.concatenate([YESimages,NOimages], axis=0)

print(images.shape)

images=images.astype('float32')
images=images/255

Y = np.concatenate((YESimagesY, NOimagesY), axis=0)
Y.shape

Y=np_utils.to_categorical(Y)
Y.shape

num_classes=Y.shape[1]
num_classes
IMG_HEIGHT , IMG_WIDTH , IMG_CHANNELS= 630,630 , 2

#spliting data

xtrain ,xtest, ytrain ,ytest=ms.train_test_split(images,Y,train_size=0.75, random_state=2,shuffle=True)

#designe U-net

model=Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=xtrain.shape[1:],activation='elu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),padding='same',activation='elu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(MaxPool2D(2,2))
model.add(Conv2D(128,(3,3),padding='same',activation='elu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(MaxPool2D(2,2))
model.add(Conv2D(256,(3,3),padding='same',activation='elu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(MaxPool2D(2,2))
model.add(Conv2D(512,(3,3),padding='same',activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())


model.add(Conv2D(1024,(3,3),padding='same',activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())


model.add(Conv2D(512,(3,3),padding='same',activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(256,(3,3),padding='same',activation='elu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())


model.add(Conv2D(128,(3,3),padding='same',activation='elu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),padding='same',activation='elu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding='same',input_shape=xtrain.shape[1:],activation='elu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(32,activation='elu'))
model.add(Dense(num_classes,activation='sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=40,batch_size=2)

pd.DataFrame(history.history).plot()

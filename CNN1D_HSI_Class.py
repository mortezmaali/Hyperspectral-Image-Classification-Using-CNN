# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:14:50 2024

@author: Morteza
"""

import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
import scipy.ndimage
from tensorflow.keras import utils as np_utils
import spectral
import scipy
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import SGD
from keras import backend as K

#import tensorflow_probability as tfp
import warnings
warnings.filterwarnings("ignore")
K.set_image_data_format('channels_last')


def load_data():
    data = sio.loadmat('C:/Users/Morteza/OneDrive/Desktop/YouTube/coding/HSI-datasets-fb174b21144c2248ed53fc8ece8072eba1445627/Salinas_corrected.mat')['salinas_corrected']
    labels = sio.loadmat('C:/Users/Morteza/OneDrive/Desktop/YouTube/coding/HSI-datasets-fb174b21144c2248ed53fc8ece8072eba1445627/Salinas_gt.mat')['salinas_gt']
    return data, labels

def splitTrainTestSet(X, y, testRatio=0.25):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=345, stratify=y)
    return X_train, X_test, y_train, y_test

def oversampleWeakClasses(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts  
    newX = X[y == uniqueLabels[0], :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y== label,:].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :]
    newY = newY[rand_perm]
    return newX, newY

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createPatches(X, y, windowSize=1, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def AugmentData(X_train):
    for i in range(int(X_train.shape[0]/2)):
        patch = X_train[i,:,:,:]
        num = random.randint(0,2)
        if (num == 0):
            flipped_patch = np.flipud(patch)
        if (num == 1):
            flipped_patch = np.fliplr(patch)
        if (num == 2):
            no = random.randrange(-180,180,30)
            flipped_patch = scipy.ndimage.interpolation.rotate(patch, no,axes=(1, 0),reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=False)    
    patch2 = flipped_patch
    X_train[i,:,:,:] = patch2
    return X_train


def Patch(data,height_index,width_index):
    #transpose_array = data.transpose((2,0,1))
    #print transpose_array.shape
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch


numComponents = 30
windowSize = 1
testRatio = 0.25
PATCH_SIZE = 1
X, y = load_data()
uniqueLabels, labelCounts = np.unique(y, return_counts=True)
print(labelCounts)
print(uniqueLabels)

X,pca = applyPCA(X,numComponents=numComponents)
XPatches, yPatches = createPatches(X, y, windowSize= 1)
X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, testRatio)
X_train, y_train = oversampleWeakClasses(X_train, y_train)
X_train = AugmentData(X_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[3])
y_train = np_utils.to_categorical(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)

model = Sequential() 
model.add(Conv1D(filters=20, kernel_size=3, activation='relu', input_shape = (30,1), padding ='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu')) 
model.add(Dense(16, activation='softmax'))

sgd = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)
ground_truth = spectral.imshow(classes = y,figsize =(5,5))



height = y.shape[0]
width = y.shape[1]

print(height,width)

patches = []
positions = []
outputs = np.zeros((height,width))

for i in range(height):
    for j in range(width):
        target = int(y[int(i+PATCH_SIZE/2), int(j+PATCH_SIZE/2)])
        if target == 0:
            continue
        else:
            image_patch = Patch(X, i, j)
            patches.append(image_patch.reshape(1, image_patch.shape[2], image_patch.shape[0]).astype('float32'))
            positions.append((int(i+PATCH_SIZE/2), int(j+PATCH_SIZE/2)))

patches = np.concatenate(patches, axis=0)
predictions = model.predict(patches)
for prediction, position in zip(predictions, positions):
    outputs[position[0]][position[1]] = np.argmax(prediction) + 1
    

predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(5,5))
#####################################################
# 
# This code is based on the tutorial at:
# https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
#
# The blog post goes over using keras+tensorflow to train and predict cats vs dogs vs pandas in images
# 
#####################################################

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import random
import pickle
import cv2
import os

import sys

def exp2(tmp_num):
    return np.power(tmp_num,2)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
#####(trainX, testX, trainY, testY) = train_test_split(data, labels_ohe, test_size=0.25, random_state=42)
num_sample=100
data_x = np.zeros(num_sample)
data_y = np.zeros(num_sample)
for itmp in range(num_sample):
   #data_x[itmp] = random.randint(1,100)
   data_x[itmp] = random.random()
   data_y[itmp] = exp2(data_x[itmp]) + (random.random()-0.5)*0.1
   #print(itmp,data_x[itmp],data_y[itmp])

(trainX, testX, trainY, testY) = train_test_split(data_x, data_y, test_size=0.25, random_state=42)

print(trainX)
print(trainY)

#print('trainY[:10]')
#print(trainY[:10])

#print("Forced System Exit!")
#sys.exit()

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
##lb = LabelBinarizer()
##trainY = lb.fit_transform(trainY)
##testY = lb.transform(testY)

# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
#model.add(Dense(1, input_shape=(1,), activation="linear"))
model.add(Dense(100, input_shape=(1,), activation="relu"))
#model.add(Dense(50, activation="relu"))
#model.add(Dense(25, activation="relu"))
#model.add(Dense(5, activation="relu"))
model.add(Dense(1))

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.5
#EPOCHS = 75
EPOCHS = 1000

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
#model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])
#model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])

# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, verbose=2)
'''
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
'''
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
x_func = np.arange(0, 1.0, 0.01)
plt.style.use("ggplot")
plt.figure()
'''
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
'''
plt.plot(data_x, data_y, 'bo')
plt.plot(x_func, exp2(x_func), 'g')
plt.plot(x_func, model.predict(x_func), 'r')
plt.savefig('tmp_fig.png')

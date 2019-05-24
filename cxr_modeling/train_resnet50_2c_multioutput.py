from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import GlobalMaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger
from imutils import paths
from batch_data_generator import cxr_data_generator as cdg
import random
#import pickle
import cv2
import os

import sys

### create base_model using ResNet50 trained on imagenet then cut off the average pooling and fully connected layers (using include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(1024,1024,3))
#base_model = ResNet50(weights='imagenet')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('bn5c_branch2c').output)


### Take base_model, then add a GlobalMaxPooling layer followed by a fully connected layer with 2 output nodes
output_pool = GlobalMaxPooling2D()(base_model.output)
predictions_1 = Dense(1, activation = 'sigmoid', name = 'predictions_1')(output_pool)
predictions_2 = Dense(1, activation = 'sigmoid', name = 'predictions_2')(output_pool)

#create graph of new whole model
whole_model = Model(inputs = base_model.input, outputs = [predictions_1, predictions_2])

for layer in whole_model.layers:
   if (layer.name != 'predictions_1') and (layer.name != 'predictions_2'):
      layer.trainable = False


# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.5
#EPOCHS = 75
#EPOCHS = 5
EPOCHS = 50
#EPOCHS = 100
#EPOCHS = 4000


# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"predictions_1": "binary_crossentropy",
	"predictions_2": "binary_crossentropy",
}
lossWeights = {"predictions_1": 1.0, "predictions_2": 1.0}

class_weight_list = { 'predictions_1': {0: 1.0 , 1: 10.0} , 'predictions_2': {0: 1.0 , 1: 51.0} }

opt = SGD(lr=INIT_LR)
#whole_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
whole_model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt, metrics=["accuracy"])

### print summary of the model architecture
whole_model.summary()

#trainX = np.array(trainX)
#trainY = np.array(trainY)

#data_gen = cdg()
data_gen = cdg(batch_size=16)

#filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath = "keras-cxr-save-2mo/weight_save/weights-{epoch:06d}.h5"
#mc = keras.callbacks.ModelCheckpoint('model_e{epoch:08d}.h5', period=10)
#mc = ModelCheckpoint('weights{epoch:08d}.h5', 
#                      save_weights_only=True, period=5)
whole_model.save('keras-cxr-save-2mo/model_save/whole_model_notrain.h5')

# serialize model to JSON
json_string = whole_model.to_json()
#model_json = model.to_json()
with open("keras-cxr-save-2mo/model_save/model_arch.json", "w") as json_file:
    json_file.write(json_string)

mc = ModelCheckpoint(filepath, save_weights_only=True, period=5)
csvl = CSVLogger('log_training_2mo.csv')

# train the neural network
H = whole_model.fit_generator(
   data_gen.get_batch(multi_output=1), 
   validation_data=data_gen.get_batch(multi_output=1, test_or_train='test'), 
   epochs=EPOCHS, 
   steps_per_epoch=5,
   validation_steps=5,
   class_weight=class_weight_list,
   callbacks=[mc,csvl], 
   verbose=2)


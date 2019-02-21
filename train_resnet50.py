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
from imutils import paths
import random
#import pickle
import cv2
import os

import sys

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('..\\..\\datasets\\combined_dogscats\\')))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
   '''
   # load the image, resize the image to be 32x32 pixels (ignoring
   # aspect ratio), flatten the image into 32x32x3=3072 pixel image
   # into a list, and store the image in the data list
   image = cv2.imread(imagePath)
   image = cv2.resize(image, (32, 32)).flatten()
   data.append(image)
   '''

   ### Get chosen image ready for ResNet50 arch. This involves resizing to (224,224)
   img = image.load_img(imagePath, target_size=(224, 224))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = preprocess_input(x)
   data.append(x[0])

   # extract the class label from the image path and update the
   # labels list
   label = imagePath.split(os.path.sep)[-2]
   #print("imagePath, label: ", imagePath, label)
   labels.append(label)

   ############################
   #sys.exit()
   ############################
data = np.array(data)
# scale the raw pixel intensities to the range [0, 1]
#data = np.array(data, dtype="float32") / 255.0
#print(data)
labels = np.array(labels)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
labels_ohe = lb.fit_transform(labels)
print('lb - labels_ohe[:10]')
print(labels_ohe[:10])

labels_ohe = to_categorical(labels_ohe)
print('tc - labels_ohe[:10]')
print(labels_ohe[:10])

print('len(lb.classes_)')
print(len(lb.classes_))

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data[:20000], labels_ohe[:20000], test_size=0.25, random_state=42)

print('trainY[:10]')
print(trainY[:10])

### create base_model using ResNet50 trained on imagenet then cut off the average pooling and fully connected layers (using include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False)
#base_model = ResNet50(weights='imagenet')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('bn5c_branch2c').output)


### Take base_model, then add a GlobalMaxPooling layer followed by a fully connected layer with 2 output nodes
output_pool = GlobalMaxPooling2D()(base_model.output)
predictions = Dense(2, activation = 'softmax')(output_pool)

#create graph of new whole model
whole_model = Model(input = base_model.input, output = predictions)

for layer in whole_model.layers:
   if layer.name != 'dense_1':
      layer.trainable = False


# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.5
#EPOCHS = 75
EPOCHS = 30

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
#model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
whole_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

### print summary of the model architecture
whole_model.summary()

#trainX = np.array(trainX)
#trainY = np.array(trainY)

print('data dimension: ', data.shape)
print('trainX dimension: ', trainX.shape)
print('trainY dimension: ', trainY.shape)

# train the neural network
H = whole_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32, verbose=2)

'''
#preds = model.predict(x)
preds = whole_model.predict(x)
print(preds.shape)
print(preds)

### Get Layer Weights

print('\n')
layer = whole_model.get_layer('bn5c_branch2c')
print( layer.get_config() )
print( layer.get_weights() )

print('\n')
layer = whole_model.get_layer('dense_1')
lw = layer.get_weights()
print( layer.get_config() )
print( lw[0] )
print( lw[0].shape )

### Create new model (with same weights as whole_model) that outputs the activations of base_model
#act_model = Model(inputs=base_model.input, outputs=base_model.get_layer('activation_49').output)
act_model = base_model
activs = act_model.predict(x)

print(activs.shape)


### trying to make heatmaps
heatmap = activs.dot(lw[0])
print(heatmap.shape)
print(heatmap)
hm1 = heatmap[0,:,:,0]
print(hm1)

max_hm1 = np.amax(hm1)
min_hm1 = np.amin(hm1)
print('max_hm1', max_hm1)
print('min_hm1', min_hm1)

hm1_norm = (hm1 - min_hm1) / (max_hm1 - min_hm1)
hm1_scale = hm1_norm * 255
print(hm1_scale)


#hm1_img_path = './output/hm.1.jpg'
#hm1_img = image.save_img(hm1_img_path,hm1)

plt.imshow(img)
#plt.imshow(hm1_scale, cmap='jet', interpolation='gaussian', alpha=0.75, extent=(0,224,224,0))
plt.imshow(hm1_scale, cmap='Reds', interpolation='gaussian', alpha=0.5, extent=(0,224,224,0))
#plt.imshow(hm1_scale, cmap=my_cmap, interpolation='gaussian', extent=(0,224,224,0))
#plt.imshow(hm1_scale, cmap=cmap1, interpolation='gaussian', extent=(0,224,224,0))
plt.colorbar()
plt.show()
'''

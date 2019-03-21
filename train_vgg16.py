from keras.applications.resnet50 import ResNet50
from keras.applications import VGG16
from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import GlobalMaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from keras.preprocessing.image import ImageDataGenerator

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
#imagePaths = sorted(list(paths.list_images('..\\..\\datasets\\combined_dogscats\\')))
imagePaths = sorted(list(paths.list_images('../tbportal_images/combined_dataset/')))
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

   ### Get chosen image ready for VGG16 arch. This involves resizing to (224,224)
   img = image.load_img(imagePath, target_size=(224, 224))
   #img = image.load_img(imagePath, target_size=(1024, 1024))
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
(trainX, testX, trainY, testY) = train_test_split(data[:80], labels_ohe[:80], test_size=0.25, random_state=42)

datagen = ImageDataGenerator(
   featurewise_center=False,  # set input mean to 0 over the dataset
   samplewise_center=False,  # set each sample mean to 0
   featurewise_std_normalization=False,  # divide inputs by std of the dataset
   samplewise_std_normalization=False,  # divide each input by its std
   zca_whitening=False,  # apply ZCA whitening
   shear_range = 10.0,
   brightness_range=[0.3, 0.6],
   rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
   zoom_range = 0.2, # Randomly zoom image 
   width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
   height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
   horizontal_flip=True,  # randomly flip images
   vertical_flip=False)  # randomly flip images


datagen.fit(trainX)

print('trainY[:10]')
print(trainY[:10])

### create base_model using VGG16 trained on imagenet then cut off the average pooling and fully connected layers (using include_top=False)
base_model = VGG16(weights=None, input_shape=(224, 224, 3), include_top=False)
#base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
#base_model = VGG16(weights='imagenet', input_shape=(1024, 1024, 3), include_top=False)
#base_model = ResNet50(weights='imagenet')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('bn5c_branch2c').output)


### Take base_model, then add a GlobalMaxPooling layer followed by a fully connected layer with 2 output nodes
output_pool = GlobalMaxPooling2D()(base_model.output)
'''
output_pool = Dense(512, activation = 'relu')(output_pool)
output_pool = Dense(128, activation = 'relu')(output_pool)
output_pool = Dense(64, activation = 'relu')(output_pool)
'''
output_pool = Dense(256, activation = 'relu')(output_pool)
predictions = Dense(2, activation = 'softmax')(output_pool)

#create graph of new whole model
whole_model = Model(inputs = base_model.input, outputs = predictions)
#whole_model = Model(input = base_model.input, output = base_model.output)
#whole_model = VGG16(weights='imagenet')


# Headless VGG16i has 20 layers
train_line = 16
for idx in range(len(whole_model.layers)):
   layer = whole_model.layers[idx]
   if idx < train_line:
      layer.trainable = False
   if idx >= train_line:
      layer.trainable = True
'''
for layer in whole_model.layers:
   print(layer.name)
   layer.trainable = False
   if layer.name == 'dense_1' or layer.name == 'dense_2':
      print(layer.name + ' Train!')
      layer.trainable = True
'''

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.5
#EPOCHS = 150
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
#sys.exit()
#trainX = np.array(trainX)
#trainY = np.array(trainY)

print('data dimension: ', data.shape)
print('trainX dimension: ', trainX.shape)
print('trainY dimension: ', trainY.shape)

# train the neural network
#H = whole_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32, verbose=2)
#H = whole_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=8, verbose=2)

#H = whole_model.fit_generator(datagen.flow(trainX, trainY, batch_size=8),
#   validation_data=(testX, testY), steps_per_epoch=len(trainX) // 8,
#   epochs=EPOCHS)

H = whole_model.fit_generator(datagen.flow(trainX, trainY, batch_size=8),
   validation_data=datagen.flow(testX, testY, batch_size=8), steps_per_epoch=len(trainX) // 8,
   epochs=EPOCHS, validation_steps=8)

whole_model.save('my_model.h5')

model2 = load_model('my_model.h5')
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

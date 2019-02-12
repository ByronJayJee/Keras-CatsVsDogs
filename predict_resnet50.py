from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import GlobalMaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

### create base_model using ResNet50 trained on imagenet then cut off the average pooling and fully connected layers (using include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False)
#base_model = ResNet50(weights='imagenet')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('bn5c_branch2c').output)


### Get chosen image ready for ResNet50 arch. This involves resizing to (224,224)
##img_path = '../dogscats/combined_dataset/dogs/dog.1.jpg'
img_path = '../../datasets/combined_dogscats/dogs/dog.1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

### Take base_model, then add a GlobalMaxPooling layer followed by a fully connected layer with 2 output nodes
output_pool = GlobalMaxPooling2D()(base_model.output)
predictions = Dense(2, activation = 'softmax')(output_pool)

#create graph of new whole model
whole_model = Model(input = base_model.input, output = predictions)

#compile the model
whole_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

### print summary of the model architecture
whole_model.summary()

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

#plt.imshow(img)
plt.imshow(hm1_scale, cmap='jet', interpolation='gaussian', alpha=0.25)
plt.colorbar()
plt.show()

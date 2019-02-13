from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import GlobalMaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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


'''
### write a function to see if magnitude of color vector is within normed threshold instead of using this lambda function
# Still trying to get this clipping to work

cmap = plt.cm.jet

my_cmap0 = cmap(np.arange(cmap.N))
my_cmap = cmap(np.arange(cmap.N))

# Create new colormap
#my_cmap = ListedColormap(my_cmap)
   
for i in range(0,len(my_cmap)):
   entry = my_cmap[i]
   mag2  = entry[0]*entry[0]
   mag2 += entry[1]*entry[1]
   mag2 += entry[2]*entry[2]
   mag2 /= 3.0

   min2 = 60*60/255/255
   max2 = 180*180/255/255
   #max2 = 90*90/255/255

   print('min2, max2, mag2', min2, max2, mag2)
   #if(mag2 > max2):
   if(mag2 > max2 or mag2 < min2):
      entry[3]=0.0
      my_cmap[i]=entry
      print('i, entry', i, entry)
   #print(my_cmap[i])

print('my_cmap')
for line in my_cmap:
   print(line)

# Create new colormap
my_cmap = ListedColormap(my_cmap)
my_cmap0 = ListedColormap(my_cmap0)
'''

'''
hm1_scale_clip = hm1_scale
hm1_scale_clip[hm1_scale<60]=np.nan
hm1_scale_clip[hm1_scale_clip>180]=np.nan
print(hm1_scale_clip)
'''

#hm1_img_path = './output/hm.1.jpg'
#hm1_img = image.save_img(hm1_img_path,hm1)

plt.imshow(img)
#plt.imshow(hm1_scale, cmap='jet', interpolation='gaussian', alpha=0.75, extent=(0,224,224,0))
plt.imshow(hm1_scale, cmap='Reds', interpolation='gaussian', alpha=0.5, extent=(0,224,224,0))
#plt.imshow(hm1_scale, cmap=my_cmap, interpolation='gaussian', extent=(0,224,224,0))
#plt.imshow(hm1_scale, cmap=cmap1, interpolation='gaussian', extent=(0,224,224,0))
plt.colorbar()
plt.show()

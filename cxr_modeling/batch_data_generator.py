import pandas as pd
import numpy as np
import logging
import sys
import time
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import GlobalMaxPooling2D

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - \n%(message)s')
logging.disable(logging.CRITICAL)

class cxr_data_generator:
   def __init__(self, batch_size=8):

      self.batch_size = batch_size

      self.csv_path = './'
      self.img_path= 'cxr_RonSummers/images/'

      self.df_train_ohe = pd.read_csv(self.csv_path+'cxr_train_ohe.csv')
      self.df_test_ohe = pd.read_csv(self.csv_path+'cxr_test_ohe.csv')

      #np.random.seed(1234)

   def get_batch(self, multi_output=None, test_or_train='train'):
      np.random.seed(1234)
      if test_or_train=='test':
         df = self.df_test_ohe
      else:
         df = self.df_train_ohe

      tot = len(df)
      #print('tot = %d' % (tot))

      while True:
         idx = np.random.randint(0, tot, size=self.batch_size)
         #print('idx: ', str(idx))
         img_name = np.array([df['Image Index'][i] for i in idx])

         logging.debug('idx')
         logging.debug(idx)

         logging.debug('df.head()')
         logging.debug(df.head())

         df_sub = df.iloc[idx,]
         df_sub.reset_index(inplace=True)

         logging.debug('df_sub')
         logging.debug(df_sub)

         df_sub_names = df_sub['Image Index']
         columns_to_drop = ['Image Index', 'No Finding', 'index']
         df_class_ohe = df_sub.drop(columns=columns_to_drop)

         logging.debug('df_sub_names')
         logging.debug(df_sub_names)

         logging.debug('df_class_ohe')
         logging.debug(df_class_ohe)

         img_data=[]
         img_class_data=[]

         # loop over the input images
         for imagePath in df_sub_names:

            ### Get chosen image ready for ResNet50 arch. This involves resizing to (224,224)
            #img = image.load_img(self.img_path+imagePath, target_size=(224, 224))
            img = image.load_img(self.img_path+imagePath)
            x = image.img_to_array(img)
            x  = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            img_data.append(x[0])

         for row in range(self.batch_size):
            img_class_data.append(df_class_ohe.iloc[row,].values.tolist())

         logging.debug('img_data')
         logging.debug(img_data)

         logging.debug('img_class_data')
         logging.debug(img_class_data)

         data_img_output = np.array(img_data)
         data_img_class_output = np.array(img_class_data)

         if (multi_output != None):
            logging.debug('multi_output')
            logging.debug(multi_output)

            #data_img_output=data_img_output[:,multi_output]
            #data_img_class_output=data_img_class_output[:,multi_output]
            data_img_class_output_tmp = {'predictions_1': data_img_class_output[:,0], 'predictions_2': data_img_class_output[:,1]}
            data_img_class_output = data_img_class_output_tmp
            logging.debug('data_img_output')
            logging.debug(data_img_output)
            
            logging.debug('data_img_class_output')
            logging.debug(data_img_class_output)

         #yield np.array(img_data), np.array(img_class_data)
         yield data_img_output, data_img_class_output

if __name__ == "__main__":

   cdg = cxr_data_generator(batch_size=1)
   batch=cdg.get_batch(multi_output=1)
   #batch=cdg.get_batch()
   logging.debug("next(batch)")
   logging.debug(next(batch))
   time.sleep(10)
   logging.debug("next(batch)")
   logging.debug(next(batch))


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
from keras.models import Sequential
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

print('testrun')

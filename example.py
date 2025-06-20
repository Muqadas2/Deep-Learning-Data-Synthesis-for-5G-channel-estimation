# Required Libraries
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, ReLU, InputLayer, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
import scipy.io as sio  # for reading MATLAB .mat files if needed

# Enable training (False means use pretrained model)
train_model = True

# Path to .mat files if you're using pre-generated MATLAB data
# train_data = sio.loadmat('my_trainData.mat')['trainData']
# train_labels = sio.loadmat('my_trainLabels.mat')['trainLabels']

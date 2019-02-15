import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

np.random.seed(123)  # for reproducibility

# Load pre-shuffled MNIST data into train and test sets
(data_train, labels_train), (data_test, labels_test) = mnist.load_data()

print(data_train.shape)
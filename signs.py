# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:24:47 2019

@author: ruchi
"""


import numpy as np


from cnn_utils import *
from basic_fun import *

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#Example of a picture
index = 10
#plt.imshow(X_train_orig[index])

#normalizing
X_train = X_train_orig/255.
X_test = X_test_orig/255.

#categorials 
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

_, _, parameters = model(X_train, Y_train, X_test, Y_test)


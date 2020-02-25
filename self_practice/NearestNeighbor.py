#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:04:41 2019

@author: xiangyinyu
"""

import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass
    
    def train(self, X, y):
        "X is NxD where each row is an example. y is 1-dimension of size N"
        # the Nearest Neighbor classifier simpily memorize all the training data
        self.Xtr = X
        self.ytr = y
        
    def predict(self, X):
        "X is NxD where each row is an example we wish to predict label for"
        num_test = X.shape[0]
        # let's make sure the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        
        # loop over all the test rows
        for i in range(num_test):
            # find the nearest image to the i'th test image
            # using the L1 distance (sum of absolute differences)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances) 
            # Returns the indices of the minimum values along an axis
            Ypred[i] = self.ytr[min_index]
        
        return Ypred
    
    
"""
 Q: With N examples, how fast are training and prediction?
 A: Training: O(1)
    prediction O(N)
    This is bad: we want classifiers that are fast at prediction; 
    slow for training is ok
"""
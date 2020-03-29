#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:07:36 2020

@author: xuanchenxiang
"""

import numpy as np

X = np.array([[1.1,0,1],
              [-4,-5,2],
              [2,4.3,3],
              [3,2,-4],
              [0.5,1,1],
              [1.1,2,-4]])
# N,D = X.shape
# N = 6, D = 3

b1 = np.array([1,1.2,-3.3,2])
W1 = np.array([[1,-2,3,1],
              [2,0,1,4],
              [3,2.3,-1,1]])
# D,H = W1.shape
# D = 3, H = 4


b2 = np.array([-2.9,0.5])
W2 = np.array([[1,-1.8],
               [2,  2,],
               [-2,  0],
               [5, 2.7]])
# H,C = W2.shape
# H = 4, C = 2

y = np.array([1,0,0,1,0,1])

N = X.shape[0]
D = X.shape[1]
H = W1.shape[1]
C = W2.shape[1]
loss = 0.0
reg = 0.0

# 2 layers
h1 = np.maximum(X.dot(W1) + b1, 0) # shape(6, 4)
scores = h1.dot(W2) + b2 # shape(6, 2)

# class probabilities
exp_scores = np.exp(scores)
probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True) # (N, C)(6, 2)
correct_probs = probs[np.arange(N),y]

# compute the loss: average cross-entropy loss and regularization
data_loss = np.sum(-np.log(correct_probs))/N
reg_loss = reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2)
loss = data_loss + reg_loss
print(loss)

# dscores
dscores = probs
dscores[np.arange(N), y] -= 1
dscores /= N

# scores -> h1.dot(W2) , so when we do bp, put h1.T first, put W2 after
# dW2 (H, C) (4, 2)
dW2 = h1.T.dot(dscores)
db2 = np.sum(dscores, axis=0, keepdims=True)

# dW1 (D, H) (3, 4)
dh = dscores.dot(W2.T)
dh[h1<=0] = 0
dW1 = X.T.dot(dh)
db1 = np.sum(dh, axis=0, keepdims=True)











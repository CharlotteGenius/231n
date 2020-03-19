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

y = np.array([1,0,0,1,0])

N = X.shape[0]
D = X.shape[1]
H = W1.shape[1]
C = W2.shape[1]
loss = 0.0

h1 = X.dot(W1) + b1
scores = np.maximum(h1, 0).dot(W2) + b2

correct_class_score = scores[np.arange(N), y]
correct_part = np.exp(correct_class_score)[:, np.newaxis]
sum_part = np.sum(np.exp(scores), axis = 1)[:, np.newaxis]

# dW1 (4, 3)
#------1
# =============================================================================
# m1 = X.dot(W2)
# m1[h1<0]=0
# print(m1)
# =============================================================================
#------2
m2 = np.zeros(h1.shape)
m2[h1>0]=1
m2=X.T.dot(m2).dot(W2)
print(m2)




# dW2 (4, 2)









#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:06:53 2020

@author: xuanchenxiang
"""

import numpy as np

X = np.array([[1,0,1],
              [4,5,2],
              [2,4,3],
              [3,2,4],
              [0,1,1]])

W = np.array([[1,2,3,1],
              [2,1,1,2],
              [1,2,1,1]])

y = np.array([1,0,3,2,0,5,6,4,7,8])

N = X.shape[0]
D = X.shape[1]
C = W.shape[1]
loss = 0.0

for i in range(N):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    correct_part = np.exp(correct_class_score)
    
    sum_part = 0
    for j in range(C):
        sum_part += np.exp(scores[j])
    
    margin = -np.log(correct_part/sum_part) # margin: Li
    loss += margin
  

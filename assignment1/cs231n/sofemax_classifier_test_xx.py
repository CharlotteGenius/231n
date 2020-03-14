#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:06:53 2020

@author: xuanchenxiang
"""

import numpy as np

X = np.array([[1.0,0.0,1.0],
              [4.0,5,2],
              [2,4,3],
              [3,2,4],
              [0,1,1]])

W = np.array([[1.0,2,3,1],
              [2,1,1,2],
              [1,2,1,1]])

y = np.array([1,0,3,2,0])

N = X.shape[0]
D = X.shape[1]
C = W.shape[1]
loss = 0.0
dW = np.zeros_like(W)

# =============================================================================
# for i in range(N):
#     scores = X[i].dot(W)
#     correct_class_score = scores[y[i]]
#     
#     correct_part = np.exp(correct_class_score)
#     print(correct_part)
#     sum_part = np.sum(np.exp(scores))
#     print(sum_part)
#     for j in range(C):
#         dW[:, j] += np.exp(scores[j])*X[i]
#     
#     margin = -np.log(correct_part/sum_part) # margin: Li
#     loss += margin
#     
#     dW /= sum_part
#     dW[:, y[i]] -= X[i]
# print(loss, dW)
# =============================================================================

# =============================================================================
# for i in range(N):
#     scores = X[i,:].dot(W)
#     probabilities = np.exp(scores)/np.sum(np.exp(scores))
#     
#     loss -= np.log(probabilities[y[i]])
#     
#     gradient_q = probabilities.reshape(1,-1)
#     gradient_q[0, y[i]] += -1
#     
#     dW += X[i,:].reshape(-1,1).dot(gradient_q)
# print(loss, dW)
# =============================================================================


for i in range(N):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    correct_part = np.exp(correct_class_score)
    sum_part = np.sum(scores)
    print(correct_part,sum_part,correct_part/sum_part)
    loss += np.log(correct_part/sum_part)
    print(loss)
    
    dW[:, y[i]] -= X[i]
    for j in range(C):
        dW[:, j] += (1/sum_part)*np.exp(scores[j])*X[i]
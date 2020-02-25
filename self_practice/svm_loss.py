#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:13:21 2019

@author: xiangyinyu
"""

import numpy as np

def L_i_vectorized(x, y, W):
    scores = W.dot(x)
    print(scores)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

x = np.array([1,2,3,4])
W = np.array([[1,2,0,1],
             [0,0,2,1],
             [2,0,1,1]])
y = np.array([1,2,3])

dW = np.zeros_like(W)
print(dW)
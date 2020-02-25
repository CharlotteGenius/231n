#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:55:12 2019

@author: xiangyinyu
"""

import numpy as np


A_tr = np.array([[0,1,2,3],
                [4,2,1,1],
                [3,3,2,0],
                [5,1,4,2],
                [3,6,0,1],
                [2,2,3,4]
                ])

A_folds = np.array_split(A_tr,3)
print(A_folds)
print(type(A_folds))

# =============================================================================
# for f in range(3):
#     A_test = np.delete(A_folds,f,axis=0)
#     A_fold = A_test[0]
#     for i in A_test[1:]:
#         A_fold = np.append(A_fold,i,axis=0)
#     print(A_fold)
#     print(type(A_fold))
# =============================================================================

# =============================================================================
# for f in range(3):
#     # f is the test fold number
#     A_test = np.delete(A_folds,f,axis=0)
#     for i in range(3):
#         if i!=f:
#             a = np.concatenate((a,A_folds[i]))
# =============================================================================


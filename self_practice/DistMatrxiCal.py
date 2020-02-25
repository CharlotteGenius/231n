#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:20:48 2019

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

A = np.array([[1,1,2,1],
              [0,1,3,2],
              [2,4,1,0]
        ])

# elementwise square
A_tr_sqr = A_tr**2
A_sqr = A**2


# The distance matrix should satisfy:
# The column number should be number of train sets, which is 6
# The row number should be number of test sets, which is 3

# 1. observe the product result, test transpose
Prod1 = np.dot(A_tr, A.T)

print('train*test_T:\n',Prod1)

# 2. observe the product result, train transpose
Prod2 = np.dot(A, A_tr.T)

print('test*train_T:\n', Prod2)

# The second product meets out expectation -- 3x6 matrix
"""
test*train_T:
 [[ 8  9 10 16 10 14]
 [13  7  9 17  8 19]
 [ 6 17 20 18 30 15]]
"""

print('train square:\n',A_tr_sqr)
print('test square:\n',A_sqr)

"""
np.sum(A_tr_sqr,axis = 1) = array([14, 22, 22, 46, 46, 33])
np.sum(A_sqr,axis = 1) = array([ 7, 14, 21])

We can see that the distance between 
the first train and the first test should be:
    14 + 7 - 2*8 = 5
the second train and the first test should be:
    22 + 7 - 2*9 = 11

What I calculate with hand is:
    5  11   9  21  33  12
    2  22  16  26  44  9
    23  9   3  22   7  24

"""

print('Dist matrix:')
Dist = np.sum(A_tr_sqr,axis = 1) + np.sum(A_sqr,axis = 1)[:,np.newaxis] - 2*Prod2
print(Dist)
"""
Dist matrix:
[[ 5 11  9 21 33 12]
 [ 2 22 18 26 44  9]
 [23  9  3 31  7 24]]

Yeah!!
"""






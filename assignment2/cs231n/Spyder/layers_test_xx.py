#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:36:16 2020

@author: xuanchenxiang
"""

import numpy as np

x = np.array([
                [[1,0,1],
                 [4,5,2]
                 ],
                [[2,4,3],
                 [3,-2,4]
                 ],
                [[0,1,1],
                 [3,2,1]
                 ],
                [[4,1,2],
                 [-1,4,3]
                 ],
                [[3,2,-3],
                 [0,1,1]
                 ]
                ])
#shape (N, d_1, d_2) (5,2,3)
# N=5 examples, each example's shape of (2,3)
# x.shape[0] = 5
# x.shape[1] = 2
# x.shape[2] = 3
# reshape each input into a vector of dimension D = d_1 * ... * d_k, and
#    then transform it to an output vector of dimension M.

w = np.array([[1.3, 2],[2.1, 1.5],[1, 2.5],[0.5, 1.2],[1.5, 2.0],[0.2, 0.3]])
# shape (D, M) (6, 2)

b = np.array([0.1, 0.3])
# shape (M,)


y = np.array([1,0,3,2,0])


#inp = x.reshape(x.shape[0], -1)
inp = np.reshape(x, (x.shape[0],-1)) # shape (5,6)
# two ways are both correct

out = np.dot(inp, w) + b
# shape (5,2)

print(out)



# dout (5,2)
# dx (5,2,3)
# dw (6,2)
# db (2,)
#----------------------------------------
#db = np.sum(dout, axis = 0)
#dw = np.dot(inp.T, dout)
#dx = np.dot(dout, w.T).reshape(x.shape)




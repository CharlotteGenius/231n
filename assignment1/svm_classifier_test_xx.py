# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

# =============================================================================
# with loops
# =============================================================================

dW = np.zeros(W.shape)
# =============================================================================
# 
# for i in range(N):
#     print('i =',i)
#     scores = X[i].dot(W)
#     correct_class_score = scores[y[i]]
#     for j in range(C):
#         print('j =',j)
#         if j == y[i]:
#             print('correct class y[{}]={}'.format(i,j))
#             continue
#         margin = scores[j] - correct_class_score + 1 # note delta = 1
#         if margin > 0:
#             loss += margin
#             dW[:, y[i]] = dW[:, y[i]] - X[i]
#             dW[:,j] = dW[:,j] + X[i]
#             print("correct class column {} substract X[{}] {}".format(y[i],i,X[i]))
#             print("current j column {} add X[{}] {}".format(j, i,X[i]))
#             print(dW,'\n')
# 
# print('L:', loss, '\ndW:\n', dW)
# =============================================================================

# =============================================================================
# without loop
# =============================================================================

# =============================================================================
# score = X.dot(W)
# print("score:\n",score)
# 
# sy = score[np.arange(N), y][:, np.newaxis]
# 
# print('correct score:\n', sy)
# 
# margin = score-sy+1
# print('margin:\n', margin)
# 
# margin = np.maximum(0, margin)
# print('margin positive:\n', margin)
# 
# margin[np.arange(N), y] = 0
# print('margin correct is zero:\n', margin)
# 
# loss = margin.sum()
# print('loss:\n', loss)
# 
# 
# 
# 
# # test: code by jariasf
# margin[margin > 0] = 1
# print('1s margin:\n', margin)
# 
# valid_margin_count = margin.sum(axis=1)
# print('count:\n',valid_margin_count)
# 
# # Subtract in correct class (-sy)
# margin[np.arange(N),y ] = -valid_margin_count
# print('margin:\n',margin)
# 
# dW = (X.T).dot(margin)
# =============================================================================

# =============================================================================
# # test: code by bruceoutdoors
# incorrect_count = np.sum(margin > 0, axis=1)
# print('incorrect_count\n',incorrect_count)
# for k in range(C):
#     print('k =',k)
#     
#     
#     wj = np.sum(X[margin[:, k] > 0], axis=0)
#     print(X[margin[:, k] > 0],'\nwj:\n',wj)
#     
#     
#     wy = np.sum(incorrect_count[y == k][:, np.newaxis] * X[y == k], axis=0)
#     print('\n1',incorrect_count[y == k][:, np.newaxis],'\n2',X[y == k],'\nwy:\n',wy)
#     dW[:, k] = wj - wy
#     print('dW[:, k]\n',dW[:, k])
# 
# print('dW:\n', dW)
# =============================================================================

#y_pred = np.max(X.dot(W), axis = 1)
#y_pred = np.argmax(X.dot(W), axis = 1)

indices = np.random.choice(np.arange(10), size = 2)
indices_choice = np.arange(10)
for it in range(8):
    
    y_batch = y[indices]
    print('in',indices)
    indices_choice = np.setdiff1d(indices_choice, indices)
    if len(indices_choice) < 1:
        break
    print('',indices_choice)
    indices = np.random.choice(indices_choice, size = 2)
    
    




import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  C = W.shape[1] # num of classes
  N = X.shape[0] # num of training data sets
  loss = 0.0
  for i in range(N):
    scores = X[i].dot(W) # the score for class i (a number)
    correct_class_score = scores[y[i]]
    for j in range(C):
      if j == y[i]:
        # y[i] is the correct class, if correct, go to next 'for' run
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # Because L_i = sum(max(0, margin),j!=yi)
        dW[:,   j] += X[i]
        dW[:,y[i]] -= X[i]
        # dmargin/dW = (dscores[j] - dscores[y[i]])/dW = d(XiWj - XiWyi)/dW
        # dmargin/dWj = Xi
        # dmargin/dWyi = -Xi

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= N
  dW /= N

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W # Notice that dW has the same shape of W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as 
  .
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  C = W.shape[1]
  N = X.shape[0]
  scores = X.dot(W)
  correct_class_score = scores[np.arange(N), y][:, np.newaxis]
  margin = np.maximum(0, scores - correct_class_score + 1)
    # np.maximum(0, x) will compare every element with 0 and keep the larger element.
  margin[np.arange(N), y] = 0
    # the correct category scores 0, because we don't plus it in the loss
  loss = margin.sum()
  loss /= N
  loss += reg*np.sum(W*W)

  # Gradient of loss
  m = np.zeros(margin.shape)
  m[margin>0]=1
  count = np.sum(m, axis=1)
  m[np.arange(N),y]=-count
  dW = (X.T).dot(m)
  dW /= N
  dW += reg*2*W

  return loss, dW





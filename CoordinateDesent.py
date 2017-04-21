# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:24:00 2016

@author: 颖
"""
import numpy as np
import pandas as pd
import math

# read data
data = pd.read_csv(‘H:\cse 250b\wine.data.txt', header = None)
# shuffle data
data = data.iloc[np.random.permutation(len(data))]
                 
# train and test data
train = data[0:128]
train_X = train.ix[:,1:13]
train_y = train.ix[:,0]
test = data[129:]
test_X = test.ix[:,1:13]
test_y = test.ix[:,0]

from sklearn.linear_model import LogisticRegression
multi_logistic = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', max_iter = 10000000, C = 100000)
multi_logistic.fit(train_X, train_y)
score_star = multi_logistic.score(test_X, test_y)
w_star = multi_logistic.coef_
intercept_star = multi_logistic.intercept_
print(score_star)

# combine intercept 
w_star = np.insert(w_star, 0, 1, axis = 1)
for i in range(3):
    w_star[i,0] = intercept_star[i]

# process train data for further functions
train_X.insert(0,0,1) # insert intercept column
train_X = train_X.as_matrix()
train_y = train_y.as_matrix()

# process test data for further functions
test_X.insert(0,0,1) # insert intercept column
test_X = test_X.as_matrix()
test_y = test_y.as_matrix()

# function to calculate training loss
def loss_func(w_i, X, y, w, index = None):
    loss = np.zeros(shape = (len(y)))
    if index != None:
      w[math.floor(index/14), math.fmod(index, 14)] = w_i
    for i in range(len(y)):
        x_i = X[i,:]
        y_i = y[i]
        neg_prob = (np.exp(np.dot(w[0],x_i)) + np.exp(np.dot(w[1],x_i)) + np.exp(np.dot(w[2],x_i)))/(np.exp(np.dot(w[y_i-1],x_i)))
        loss[i] = np.log(neg_prob)
    return sum(loss) 
    
L_star = loss_func(1,train_X, train_y, w_star)
L_star

# normalize train_X
from sklearn import preprocessing
train_X_norm = preprocessing.normalize(train_X, norm='l2')

# function to calculate test error
def test_error(Xtest, ytest, w):
    predict = np.zeros(shape = len(ytest))
    for i in range(len(ytest)):
        max_j = 2
        max_prob = np.dot(w[2], Xtest[i])
        for j in range(2):
            prob = np.dot(w[j], Xtest[i])
            if prob > max_prob:
                max_prob = prob
                max_j = j
        predict[i] = max_j
    return 1 - np.sum((predict+1) == ytest)/len(ytest)
    
# initialize w
w_0 = np.random.rand(3,14)
w_0

# function to calculate gradient of an coordinate
def gradient(X, y, w, index):
  delta = 0.0001
  loss1 = loss_func(0, X, y, w)
  w[math.floor(index/14), math.fmod(index, 14)] -= delta
  loss2 = loss_func(0, X, y, w)
  return (loss1 - loss2) / delta

# function to set calculate new w_i, parameters: index of w_i
def new_wi(index, X, y, w):
  step = -0.1 * np.sign(gradient(X, y, w, index))
  while abs(step) > 0.0001:
    tmp_loss = loss_func(X, y, w)
    w[math.floor(index/14), math.fmod(index, 14)] += step# * gradient(X, y, w, index) 
    if loss_func(X, y, w) > tmp_loss:
      w[math.floor(index/14), math.fmod(index, 14)] -= step# * gradient(X, y, w, index)
      step /= 2
  return w

from scipy.optimize import fmin
def new_w_i(index, old_wi, X, y, w):
    new_wi = fmin(loss_func, args = (X, y,w,index), x0 = old_wi, disp = False)
    return new_wi
    
# random-feature coordinate descent
import warnings
warnings.filterwarnings('ignore')
from random import randint
loss = []
w = w_0
t = 0
tmp_loss = 1
while tmp_loss > 0.1:
    index = randint(0,41)
    old_w_i = w[math.floor(index/14), math.fmod(index, 14)]
    new_wi = new_w_i(index, old_w_i, w, train_X, train_y)[0]
    w[math.floor(index/14), math.fmod(index, 14)] = new_wi
    tmp_loss = loss_func(new_wi, train_X, train_y, w) # update loss
    loss.append(tmp_loss)
    print(t)
    t = t+1

# my coordinate descent
def select_coordinate(X, y, w):# question (a)
    grad = np.zeros(42)
    for i in range(42):
        grad[i] = gradient(X, y, w, i)
    max_i = np.argmax(abs(grad))
    return max_i

from sklearn import preprocessing
train_X_norm = preprocessing.normalize(train_X, norm='l2')
import math
    
Loss= []
Error = []
w = w_0
t2 = 0
tmp_loss = 10000
while tmp_loss > 0.01:
    index = select_coordinate(train_X_norm, train_y, w)
    old_w_i = w[math.floor(index/14), math.fmod(index, 14)]
    new_wi = new_w_i(index, old_w_i, train_X_norm, train_y, w)[0] # new value of w_i
    w[math.floor(index/14), math.fmod(index, 14)] = new_wi # update w
    tmp_loss = loss_func(0, train_X_norm, train_y, w) # update loss
    Loss.append(tmp_loss)
    Error.append(test_error(test_X, test_y, w))
    if math.fmod(t2, 100) == 0: print(tmp_loss)
    t2 = t2 + 1






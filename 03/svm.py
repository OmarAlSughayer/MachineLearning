# CSE446 HW03 - Question 03
# Omar Adel AlSughayer (1337255)

from numpy import *
from scipy import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import time

# do one pass of SGD over the dataset.
def svm_binary(predicate, data, lamb=1e-2, eta=1e-2, w=None):
    (labels, features) = data
    if not w:
        w = zeros(features.shape[1] + 1)
    # iterate over all  rows of features, i.e. all points (xi, yi)
    for i in range (0, len(labels)):
        # one row of features 
        x = features[i,:]
        # change the label to a boolean in 1 or 0, then change to 1 or -1
        y = 2*int(predicate(labels[i])) - 1

        # the indicator function 
        indi = int(1.0 - y*(np.dot(w[1:], x) + w[0]) > 0.0)

        # update weights
        # w = w - eta *[w0 = -1*y*indi, w = lamb*w - y*indi*x]
        w = w - eta*np.append(-1*y*indi, lamb*w[1:] - y*indi*x)
 
    return w

    '''
    # basically the same as above but a bit more readable
    # iterate over all the rows of features, i.e. all points
    for i in range (0, len(labels)):
        # add a dummy variable to x to multiply it by w0
        x = np.append([0], features[i,:])
        # change the label to a boolean in 1 or 0, then change 0 to -1
        y = int(predicate(labels[i]))
        if (y == 0):
            y = -1
        # get w0
        w0 = w[0]
        # the indicator function 
        indi = int(1.0 - y*(np.dot(w, x) + w0) > 0.0)

        # new w0
        nw0 = -1*y*indi
        # new weights
        nw = lamb*w - y*indi*x
        # new weights vector
        nw[0] = nw0
        #update weights
        w = w - eta*nw
 
    return w
    '''

def svm_accuracy(predicate, w, data):
    (labels, features) = data
    error = 0
    for i in xrange(labels.shape[0]):
        xi = features[i, :]; yi = 2 * predicate(labels[i]) - 1
        if (w[0] + w[1:].dot(xi)) * yi < 0:
            error += 1
    return 1 - error/(1.0 * labels.shape[0])

# iterates over @arg{params} and return that which attains the maximum
# accuracy on the validation set.
def svm(predicate, train, validation, params=[(1e-2, 1e-2)]):
    max_accuracy = 0.; max_param = None; max_w = None
    
    # for all pairs of (lamb, eta)
    for i in range(0, len(params)):
        (lamb, eta) = params[i]
        curr_w = svm_binary(predicate, train, lamb, eta)
        curr_accuracy = svm_accuracy(predicate, curr_w, validation)

        # check against maximum accuracy
        if(curr_accuracy > max_accuracy):
            max_accuracy = curr_accuracy
            max_param = params[i]
            max_w = curr_w

    return (max_w, max_param, max_accuracy)

# returns the predicted class of x using the one-versus-all scheme.
def svm_predict(x, svms):
    def svm_predict_kernel(x, svms):
        # svms = [*(l, w_l)]
        # x = (6912,)
        max_hh = 0; max_label = -1
        
        for i in range(0, len(svms)):
            (label, w_l) = svms[i]
            curr_hh = (w_l[0] + w_l[1:].dot(x))
            if curr_hh > max_hh:
                max_hh = curr_hh
                max_label = label

        return max_label

    if len(x.shape) > 1:
        predict = zeros(x.shape[0], dtype=int64)
        for ii in xrange(x.shape[0]):
            predict[ii] = svm_predict_kernel(x[ii, :], svms)
        return predict
    else:
        return svm_predict_kernel(x, svms)

# Computes one-vs-all classifiers by iterating over @arg{labels}
# returns a set of tuples, weights = [*(l, w_l)]
def svm_multiclass(train, validation, params=[(1e-2, 1e-2)]):
    (labels, features) = train
    label_set = np.unique(labels)
    weights = []

    # iterate over all labels
    for i in range(0, len(label_set)):
        label = label_set[i]
        (w, param, acc) = svm(lambda ll : ll == label, train, validation, params)
        weights.append((label, w))

    return weights
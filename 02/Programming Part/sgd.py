# CSE446 HW02 - Question 06
# Omar Adel AlSughayer (1337255)

from math import exp
import random
import numpy as np

# Calculates logistic of a numpy matrix
def logistic(x):
    
    # the logisitc function 
    def segmoid(a):
        return 1.0/(1.0 + exp(-1*a))

    # the vectorized version of the logistic function
    segmoid = np.vectorize(segmoid)

    return segmoid(x);

# Calculates accuracy of predictions on data
def accuracy(data, predictions):

    correct_predictions = 0;

    for i in range (0, len(predictions)):
        real = data[i]['label'].item(0) == True or data[i]['label'].item(0) == 1
        guess = predictions[i].item(0) > 0.5
        if(real == guess):
            correct_predictions += 1

    return correct_predictions/float(len(predictions))

class model:
    def __init__(self, structure):
        self.weights=[]
        self.bias = []
        for i in range(len(structure)-1):
            self.weights.append(np.random.normal(size=(structure[i], structure[i+1])))
            self.bias.append(np.random.normal(size=(1, structure[i+1])))
        
    # Calculates prediction based on model
    def predict(self, point):
        a = self.feedforward(point)
        return a[len(a)-1];

    # Updates model using learning rate and L2 regularization
    def update(self, a, delta, eta, lam):
        limit = -1*int(len(self.weights) == 1)

        for i in range(len(self.weights)-1, limit, -1):
            x = a[i]
            w = self.weights[i]
            p = delta[i]
            b = self.bias[i]
            self.weights[i] = w - eta*(lam*w - x.T*p)
            self.bias[i] = np.matrix(b - eta*(b - 1*p.T))

        pass

    # Performs the forward step of backpropagation
    def feedforward(self, point):
        a = []
        a.append(point['features']) # append m0

        for i in range(0, len(self.weights)):
            mi = np.dot(a[i], self.weights[i]) + self.bias[i]
            a.append(logistic(mi))

        return a
    
    # Backpropagates errors
    def backpropagate(self, a, label):
        delta = []
        delta.append(label - a[len(a)-1])

        for i in range(len(a)-1, 1, -1):
            mi_1 = a[i-1]
            wi = self.weights[i-1]
            mi = delta[0]

            #mi_1 = np.dot(wi.T, mi) * segmoid_prime(mi_1)
            mi_1 = np.multiply(np.dot(wi, mi), (mi_1 - np.multiply(mi_1, mi_1)).T)
            delta.insert(0, mi_1)

        return delta

    # Trains your model
    def train(self, data, epochs, rate, lam):

        for i in range(epochs*len(data)):
            index = random.randint(0, len(data)-1)
            # pick a random point
            point = data[index]
            # get the true y 'label' out
            y = int(point['label'].item(0) == True)
            # make a list of the features
            features_list = self.feedforward(point)
            # approximate the true gradient using predict 
            delta = self.backpropagate(features_list, y)
            # run update with (self, features_list, delta, rate, lam)
            self.update(features_list, delta, rate, lam)

        pass

def logistic_regression(data, lam=0.0001):
    m = model([data[0]["features"].shape[1], 1])
    m.train(data, 1000, 0.05, lam)
    return m
    
def neural_net(data, lam=0.0001):
    m = model([data[0]["features"].shape[1], 15, 1])
    m.train(data, 100, 0.01, lam)
    return m

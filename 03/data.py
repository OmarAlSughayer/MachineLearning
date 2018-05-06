from numpy import *
from scipy import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import time

def get_data():
    data = load("data.npy")
    ntrain = 10000
    nvalid = 2422
    train = (data[:ntrain, 0].astype(int64), data[:ntrain, 1:])
    valid = (data[ntrain:ntrain+nvalid, 0].astype(int64), data[ntrain:ntrain+nvalid, 1:])
    test = (data[ntrain+nvalid:, 0].astype(int64), data[ntrain+nvalid:, 1:])
    return (train, valid, test)

def dump_model(model, fname):
    pickle.dump([time.time(), model], open(fname, 'wb'))

def load_model(fname):
    tstamp, model = tuple(pickle.load(open(fname, 'r')))
    return model

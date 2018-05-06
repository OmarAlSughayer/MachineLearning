from numpy import *
from scipy import *
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import itertools, pickle, time, svm, unittest, sys, data
from unittest import TestCase

class SVMTest(unittest.TestCase):
    def setUp(self):
        self.data = (self.train, self.valid, self.test) = data.get_data()
        self.params = list(itertools.product([1, 1e-1, 1e-2, 1e-3, 1e-4], [1e-1, 1e-2, 1e-3, 1e-4]))
        self.results = pickle.load(open("tests.p"))

    @unittest.skip("Comment out this line when ready.")
    def test_svm_binary(self):
        w = svm.svm_binary(lambda ll : ll == 0, self.train, lamb=1.0, eta=1e-2)
        self.assertLessEqual(la.norm(w - self.results[0]),  1e-3)

    @unittest.skip("Comment out this line when ready.")
    def test_svm(self):
        w, max_p, max_acc = svm.svm(lambda ll : ll == 0, self.train, self.valid, params=self.params)
        self.assertLessEqual(la.norm(w - self.results[1]),  1e-3)
        self.assertEqual(max_p, (1e-3, 1e-2))
        self.assertGreaterEqual(max_acc, 0.99)

    @unittest.skip("Comment out this line when ready.")
    def test_svm_predict(self):
        w, max_p, max_acc = svm.svm(lambda ll : ll == 1, self.train, self.valid, params=self.params)
        predict = svm.svm_predict(self.test[1], [(1, w)])
        self.assertGreaterEqual(sum(predict == self.test[0]), 460)

    #@unittest.skip("Comment out this line when ready.")
    def test_svm_multiclass(self):
        svms = svm.svm_multiclass(self.train, self.valid, params=self.params)
        preds = svm.svm_predict(self.test[1], svms)
        acc = mean(preds == self.test[0])
        print
        print "Your current accuracy is:", acc
        self.assertGreaterEqual(acc, .94)
        #TODO: uncomment and dump the trained model.
        data.dump_model(svms, "svms.p")

if __name__ == '__main__':
    unittest.main()

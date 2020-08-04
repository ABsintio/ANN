# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:08:00 2020

@author: ricca
"""

from util.function.functions import ActivationFunction
from util.algorithm.perceptronrule import perceptron_rule
from util.algorithm.gradientdescent import gradient_descent
import numpy as np

class Perceptron(object):
    def __init__(self, learning_rate=0.01, n_iter=100, actfun="sgn"):
        assert ActivationFunction.checkfun(actfun)
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.actfun = ActivationFunction.getfunction(actfun)
        self.w_ = None
    
    def fit(self, traning_example):
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return self
    
    def getpesi(self): 
        return self.w_
    
    def getiter(self):
        return self.n_iter
    
    def getlearningrate(self):
        return self.learning_rate
    
    def getactfun(self):
        return self.actfun
    

class PerceptronPR(Perceptron):
    """Perceptron trained by Perceptron Rule"""
    def __init__(self, learning_rate=0.01, n_iter=100):
        super().__init__(
                learning_rate=learning_rate,
                n_iter=n_iter,
                actfun="sgn")
        self.error_ = []
        
    def fit(self, traning_example):
        self.w_, self.error_ = perceptron_rule(
                traning_example, self.actfun,
                self.learning_rate, self.n_iter)
        return self
    
    def geterror(self):
        return self.error_
    
    def predict(self, X):
        return self.actfun(self.net_input(X))


class PerceptronGD(Perceptron):
    """Perceptron trained by Gradient Descent"""
    def __init__(self, learning_rate=0.01, n_iter=100):
        super().__init__(
                learning_rate=learning_rate,
                n_iter=n_iter,
                actfun="linear")
        self.costs_ = []
      
    def fit(self, traning_example):
        self.w_, self.costs_ = gradient_descent(
                traning_example, self.actfun,
                self.learning_rate, self.n_iter)
        return self
    
    def getcosts(self):
        return self.costs_
    
    def predict(self, X):
        return ActivationFunction.sgn(self.net_input(X))
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 19:54:08 2020

@author: ricca
"""

import numpy as np

class ActivationFunction(object):
    @staticmethod
    def default():
        return "Invalid activation function"
    
    @staticmethod
    def linear(y: float):
        return y
    
    @staticmethod
    def sgn(y: float):
        return np.where(y >= 0, 1, -1)
    
    @staticmethod
    def heaviside(y: float):
        return np.where(y >= 0, 1, 0)
    
    @staticmethod
    def sigmoid(y: float):
        return 1/(1 + np.exp(-y))
    
    @staticmethod
    def hypertan(y: float):
        return np.tanh(y)
    
    @staticmethod
    def getfunction(f):
        functs = {
                "linear" : ActivationFunction.linear,
                "sgn" : ActivationFunction.sgn,
                "heaviside" : ActivationFunction.heaviside,
                "sigmoid" : ActivationFunction.sigmoid,
                "hypertan" : ActivationFunction.hypertan
        }
        return functs.get(f, ActivationFunction.default)
    
    @staticmethod
    def checkfun(f):
        functs = {
                "linear" : ActivationFunction.linear,
                "sgn" : ActivationFunction.sgn,
                "heaviside" : ActivationFunction.heaviside,
                "sigmoid" : ActivationFunction.sigmoid,
                "hypertan" : ActivationFunction.hypertan
        }
        return f in functs
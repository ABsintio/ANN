# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 23:30:42 2020

@author: ricca
"""

import numpy as np


def shuffle_data(X, y, rgen):
    """Fa una permutazione dei dati"""
    r = rgen.permutation(len(y))
    return X[r], y[r]


def stochastic_gd(traning_examples, function, shuffle=True, learning_rate=0.01, n_iter=100):
    """
    Ad ogni iterazione dobbiamo andare a fare un upadate dei pesi 
    tramite la seguente regola. Siano wi il peso i-esimo, xi l'input 
    i-esimo, t il target output, o l'output ottenuto e l il lerning
    rate allora 
    
    wi = wi + l * (t - o) * xi
    o = w1*x1 + ... + wN*xN + w0
    
    :params ndarray traning_example: insieme di coppie (sample, target)
    :params function: la funzione di attivazione
    :params learning_rate: costante reale positiva
    :params n_iter: numero di iterazioni da effettuare
    :return: l'insieme dei pesi ed il vettore dei costi
    """
    X, y = [], []
    for k, v in traning_examples:
        X.append(k)
        y.append(v)
    
    X, y = np.array(X), np.array(y)
    costs = []
    w_ = np.random.uniform(-1, 1, X.shape[1] + 1)
    rgen = np.random.RandomState(1)
    
    while n_iter > 0:
        cost = []
        if shuffle:
            X, y = shuffle_data(X, y, rgen)
        for x, reference in zip(X, y):
            output = function(np.dot(x, w_[1:]) + w_[0])
            error = (reference - output)
            w_[1:] += learning_rate * error * x
            w_[0] += learning_rate * error
            cost.append(error**2 / 2.0)
        avg_cost = sum(cost) / len(y)
        costs.append(avg_cost)
        n_iter -= 1
    
    return w_, costs
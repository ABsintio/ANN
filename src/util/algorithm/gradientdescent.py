# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:29:39 2020

@author: ricca
"""

import numpy as np

def gradient_descent(traning_example, function, learning_rate=0.01, n_iter=10):
    """
    Ad ogni iterazione andiamo a fare un update dei pesi tramite la 
    seguente regola. Siano wi l'i-esimo peso, xi l'i-esimo input, T
    l'insieme di tutte le reference e O l'insieme di tutti gli output
    per tutti gli input, allora
    
    wi = wi + Delta(wi)
    Delta(wi) = learning_rate * errore
    errore = sum_{d in D}(xid * (td - od))
    
    :params ndarray traning_example: insieme di coppie (sample, target)
    :params function: la funzione di attivazione
    :params learning_rate: costante reale positiva
    :params n_iter: numero di iterazioni da effettuare
    :return: l'insieme dei pesi ed il numero di update
    """
    X, y = [], []
    for k, v in traning_example:
        X.append(k)
        y.append(v)
    
    X, y = np.array(X), np.array(y)
    costs = []
    w_ = np.random.uniform(-1, 1, X.shape[1] + 1)
    
    while n_iter > 0:
        output = function(np.dot(X, w_[1:]) + w_[0])
        errors = (y - output)
        w_[1:] += learning_rate * np.dot(X.T, errors)
        w_[0] += learning_rate * errors.sum()
        cost = (errors**2).sum() / 2.0
        costs.append(cost)
        n_iter -= 1
    
    return w_, costs

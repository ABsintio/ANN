# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 20:00:57 2020

@author: ricca
"""

import numpy as np

def perceptron_rule(traning_example, function, learning_rate=0.1, n_iter=100):
    """
    Ad ogni iterazione dobbiamo andare a fare un upadate dei pesi 
    tramite la seguente regola. Siano wi il peso i-esimo, xi l'input 
    i-esimo, t il target output, o l'output ottenuto e l il lerning
    rate allora 
    
    wi = wi + l * (t - o) * xi
    o = f(w1*x1 + ... + wN*xN + w0)
    
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
    pesi = np.zeros(X.shape[1] + 1)
    errors = []
    
    while n_iter > 0:
        error = 0
        for x, reference in zip(X, y):
            output = function(np.dot(x, pesi[1:]) + pesi[0])
            update = learning_rate * (reference - output)
            pesi[1:] = pesi[1:] + update * x
            pesi[0] = pesi[0] + update
            error += int(update != 0.0)
        errors.append(error)
        n_iter -= 1
    
    return pesi, errors
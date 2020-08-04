# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 22:28:57 2020

@author: ricca
"""

from neurons.perceptrons import PerceptronPR, PerceptronGD
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np


def plot_decision_region(X, y, classifier, resolution=0.02):
    # Setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0],
                    y=X[y==cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor="black")
    

def main():
    # Carico l'iris dataset
    dataset = pd.read_csv("datasets/iris/iris.csv")
    
    # Prendo le prime 100 righe considerando solo il valore nella colonna 4
    y = dataset.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)
    
    # Prendo le prime 100 righe considerando solo i valori dalla colonna 0 alla 2 esclusa
    X = dataset.iloc[0:100, [0, 2]].values
    
    # Plot data
    plt.scatter(X[:49, 0], X[:49, 1], color="red", marker="o", label="setosa")
    plt.scatter(X[49:100, 0], X[49:100, 1], color="blue", marker="x", label="versicolor")
    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")
    plt.legend(loc="upper left")
    plt.show()
    
    # Creo l'insieme di addestramento
    traning_examples = list(zip(X, y))
    perceptron = PerceptronPR(n_iter=10, learning_rate=0.1)
    perceptron = perceptron.fit(traning_examples)
    
    # Otteniamo i pesi, gli errori e le iterazioni totali fatte
    errors = perceptron.geterror()
    epochs = perceptron.getiter()
    
    plt.plot(range(1, epochs + 1), errors, marker='o')
    
    plt.show()
    
    plot_decision_region(X, y, classifier=perceptron)
    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")
    plt.legend(loc="upper left")
    plt.show()
    
    perceptron = PerceptronGD(n_iter=10, learning_rate=0.0001)
    perceptron = perceptron.fit(traning_examples)
    
    # Otteniamo i pesi, gli errori e le iterazioni totali fatte
    costs = perceptron.getcosts()
    epochs = perceptron.getiter()
    
    plt.plot(range(1, epochs + 1), np.log10(costs), marker='o')
    
    plt.show()
    
    plot_decision_region(X, y, classifier=perceptron)
    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")
    plt.legend(loc="upper left")
    plt.show()
    

if __name__ == "__main__":
    main()
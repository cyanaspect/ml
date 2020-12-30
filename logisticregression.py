import os
import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt
import utils


def plot_data(x, y):
    fig = plt.figure()

    pos = y == 1
    neg = y == 0

    plt.plot(x[pos, 0], x[pos, 1], "go", mec="k")
    plt.plot(x[neg, 0], x[neg, 1], "ro", mec="k")

    plt.show()


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def grad_descent(x, y, theta, alpha, num_iter):
    m = x.shape[0]
    n = x.shape[1] - 1
    theta = theta.copy()
    x = normalize(x)

    J_history = []

    for _ in range(num_iter):
        J_history.append(cost_function(x, y, theta))
        h = sigmoid(x.dot(theta.T))
        theta = theta - (alpha/m)*(h-y).dot(x)

    return theta, J_history


def normalize(x):
    m = x.shape[0]
    n = x.shape[1] - 1

    means = np.mean(x, axis=0)

    for i in range(m):
        x[i] -= means

    for i in range(1, n + 1):
        x[:, i] = x[:, i] / np.ptp(x[:, i])

    return x


def cost_function(x, y, theta):  # tested
    m = x.shape[0]
    n = x.shape[1]

    h = sigmoid(x.dot(theta.T))

    return (-1/m)*(y.T.dot(np.log(h)) + (1-y).T.dot(np.log(1-h)))


def predict(theta, x):
    return np.round(sigmoid(x.dot(theta.T)))






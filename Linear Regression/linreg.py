import pandas as pd
import numpy as np
import plotting as pl
from matplotlib import pyplot as plt


def print_weight_analysis():
    below_th = -0.25
    above_th = 0.25
    total_extr = []
    for i in range(len(idx)):
        weights = W[idx[i]]
        print("-------------------")
        print(f"Current: P={P[idx[i]]}")
        maximum = [-1, 0]
        minimum = [-1, 0]
        below = []
        above = []
        for j in range(len(weights)):
            w = weights[j]
            if w > maximum[1]:
                maximum = [j, w]
            if w < minimum[1]:
                minimum = [j, w]
            if w < below_th:
                below.append([j, w])
            if w > above_th:
                above.append([j, w])
        print("Maximum:", maximum)
        print("Minimum:", minimum)
        print(f"Below {below_th}:", below)
        print(f"Above {above_th}:", above)
        print("Total extreme:", len(above)+len(below))
        total_extr.append([P[idx[i]], len(above)+len(below)])
        i += 1
    return total_extr


def calc_error(w, x, y, delta=0):
    assert len(w) == x.shape[1] and x.shape[0] == len(y)
    sum = 0
    for i in range(0, x.shape[0]):
        sum += 0.5 * ((np.dot(w, x[i]) - y[i])**2)
    sum += 0.5 * delta * np.dot(w, w)
    return sum / len(y)


def get_errors_and_weight(train_x, train_y, test_x, test_y, delta=0):
    x_pinv = np.linalg.inv(np.transpose(train_x) @ train_x + (delta * np.identity(train_x.shape[1]))) @ np.transpose(train_x)
    w = np.dot(x_pinv, train_y)
    return calc_error(w, train_x, train_y, delta), calc_error(w, test_x, test_y, delta), w


# read in the data
# training
train_x = np.loadtxt(open("data/xtrain.csv", "r+"), delimiter=",")
train_y = np.loadtxt(open("data/ytrain.csv", "r+"), delimiter=",")

# testing
test_x = np.loadtxt(open("data/xtest.csv", "r+"), delimiter=",")
test_y = np.loadtxt(open("data/ytest.csv", "r+"), delimiter=",")

# setting the regularization term
delta = 0

# 0: 30, 1: 40, 2: 50, 3: 75, 4: 100, 5: 200, 6: 300, 7: 400, 8: 500
P = [30, 40, 50, 75, 100, 200, 300, 400, 500]
idx = [0, 1, 2, 3, 4, 8]
W = []
train_errors = []
test_errors = []
for i in range(0, len(P)):
    train_error, test_error, w = get_errors_and_weight(train_x[0:P[i]], train_y[0:P[i]],
                                                                  test_x[0:P[i]], test_y[0:P[i]], delta)
    train_errors.append(train_error)
    test_errors.append(test_error)
    W.append(w)


pl.plot_total_extreme_values(print_weight_analysis())
pl.plot_error(train_errors, test_errors, P)
pl.plot_W(W, P, idx)
plt.show()




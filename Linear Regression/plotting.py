from matplotlib import pyplot as plt
import numpy as np


def plot_error(train_err, test_err, P):
    plt.figure()
    plt.title(f"Training and Testing error for different P values")
    plt.xlabel("P")
    plt.ylabel("Training and Test Error")
    plt.plot(P, train_err, color='green', label='Training Error')
    plt.plot(P, test_err, color='red', label='Testing Error')
    plt.legend()


def plot_W(W, P, idx):
    ticks = np.arange(len(W[0]))
    for i in idx:
        plt.figure()
        plt.title(f"Weights for P={P[i]}")
        plt.xlabel('$w_{i}$')
        plt.ylabel("Weight Value")
        plt.xticks(ticks)
        plt.bar(ticks, W[i], color='green', align='center')


def plot_total_extreme_values(vals):
    plt.figure()
    plt.title("Total extreme weight values for different P")
    plt.xlabel('P')
    plt.ylabel("Amount of extreme values")
    plt.plot([val[0] for val in vals], [val[1] for val in vals], color="green")

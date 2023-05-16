import random
import math
import numpy as np
import pandas as pd
from plotting import *
from matplotlib import pyplot as plt
from calculation import *


def main():
    # Reading in the Data File
    df = pd.read_csv("data/lvqdata.csv", sep=",")

    # Determining the size of the input vectors and the amount of vectors
    amt_points, input_dim = df.shape

    # Adding label to the data frame
    labels = np.concatenate((np.full((50, 1), 0, dtype=int), np.full((50, 1), 1, dtype=int)), axis=None)
    df["Label"] = labels

    # Setting learning variables
    learning_rate = 0.001
    max_learning_rate = 0.02
    learning_rate_increase = 0.001
    t_max = 100
    threshold = 10
    amt_protos = 1  # adjust for different prototype amount per class
    amt_subsets = 10  # adjust for different M of the cross validation
    seed = 1
    random.seed(seed)

    # Creating the subsets of the data used in the cross validation
    # In the case that the data frame cannot be divided equally into
    # M subsets then the amount of points n in one subset is rounded up
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = math.ceil(df.shape[0] / amt_subsets)
    list_df = [df[i:i+n] for i in range(0, df.shape[0], n)]

    # Data structures to store the different values during the cross validation
    # The error lists will contain tuples (x, y) where x = mean, y = std dev
    # The index i of the lists represents the tuple corresponding to i + 1 prototypes
    training_error_list = []
    testing_error_list = []
    prototype_list = []
    prototype_trace_list = []
    learning_rate_list = []

    while learning_rate <= max_learning_rate:
        print(f"Learning Rate = {learning_rate}")
        training_error, test_error, prototypes, prototype_trace = \
            cross_validation(list_df, amt_subsets, amt_protos, t_max, learning_rate)
        training_error_list.append(training_error)
        testing_error_list.append(test_error)
        prototype_list.append(prototypes)
        prototype_trace_list.append(prototype_trace)
        learning_rate_list.append(learning_rate)
        learning_rate += learning_rate_increase

    print(training_error_list)
    print(testing_error_list)

    ticks = [0.001, 0.005, 0.01, 0.015, 0.02]
    plot_curve(training_error_list, testing_error_list, 0, x_data=learning_rate_list, x_lab="Learning Rate Eta",
               ticks=ticks ,y_lab="Error", title=f"Mean Training and Testing Error for M={amt_subsets}")
    plot_curve(training_error_list, testing_error_list, 1, x_data=learning_rate_list, x_lab="Learning Rate Eta",
               ticks=ticks, y_lab="Error", title=f"Standard Deviation of Training and Testing Error for M={amt_subsets}")
    plt.show()


main()

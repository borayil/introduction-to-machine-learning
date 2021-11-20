import random
import numpy as np
import pandas as pd
from plotting import *
from calculation import *


# Reading in the Data File
df = pd.read_csv("lvqdata.csv", sep=",")

# Determining the size of the input vectors and the amount of vectors
amt_points, input_dim = df.shape

# Adding label to the data frame
labels = np.concatenate((np.full((50, 1), 0, dtype=int), np.full((50, 1), 1, dtype=int)), axis=None)
df["Label"] = labels

# Setting learning variables
learning_rate = 0.002
t_max = 100
threshold = 10
amt_protos = 1 # adjust for different prototype amount per class
random.seed(123)

# Creating the prototypes
# prototypes is also a pandas dataframe
prototypes, prototype_trace = create_prototypes_random(amt_protos, df)

# Algorithm
# epoch loop
training_errors = []
count = 0
for i in range(0, t_max):
    for index, row in df.iterrows():
        # randomize
        df = df.sample(frac=1).reset_index(drop=True)
        # for current point check closest prototype
        closest_prototype, closest_idx = get_closest_prototype(prototypes, row)
        # update the prototype
        function = (lambda x, y: x + y) if int(closest_prototype['Label']) == int(row['Label']) else (lambda x, y: x - y)
        closest_prototype['X'] = function(closest_prototype['X'], learning_rate * (row['X'] - closest_prototype['X']))
        closest_prototype['Y'] = function(closest_prototype['Y'], learning_rate * (row['Y'] - closest_prototype['Y']))
        prototype_trace.iloc[closest_idx]['trace_X'].append(closest_prototype['X'])
        prototype_trace.iloc[closest_idx]['trace_Y'].append(closest_prototype['Y'])
    training_error = calc_training_error(df, prototypes)
    if i != 0 and training_error == training_errors[i - 1]:
        count += 1
    else:
        count = 0
    if count == threshold:
        break
    training_errors.append(training_error)

plot_data_points(df, prototypes, prototype_trace)
plot_learning_curve(training_errors)


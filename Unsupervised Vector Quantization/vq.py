import random
import numpy as np
import pandas as pd
from plotting import *
from calculation import *
from matplotlib import pyplot as plt

# Reading in the Data File
df = pd.read_csv("data/simplevqdata.csv", sep=",")

# Determining the size of the input vectors and the amount of vectors
amt_points, input_dim = df.shape

# Setting learning variables
learning_rate = 0.004
t_max = 15
amt_protos = 1  # adjust for different prototype amount per class
max_protos = 5
seed_val = 31
random.seed(seed_val)
np.random.seed(seed_val)

# Index i of this holds the Hvq for epoch i.
quantization_errors = []
final_hvq = []
final_protos = []
final_proto_trace = []

print("\tStarting Vector Quantization!")
print("Parameters")
print("K = " + str(amt_protos))
print("tmax = " + str(t_max))
print("Learning Rate = " + str(learning_rate))
print("Seed = " + str(seed_val))

while amt_protos <= max_protos:
    # Creating the prototypes change the function to initialize in a different way
    # prototypes is also a pandas dataframe
    prototypes, prototype_trace = create_prototypes_random(amt_protos, df)
    # reset values for this iteration
    min_error = float('inf')
    min_error_prototypes = None
    min_error_trace = None
    quantization_errors.append([])
    for i in range(0, t_max):
        # Make a copy of df before each epoch to allow for random selection of points in each epoch, and to keep df same
        df_copy = df.copy()

        # Show progress on command line
        print('\r', "Epoch = " + str(i), end='')
        for j in range(amt_points):
            # Get single example
            drop_index = np.random.choice(df_copy.index, 1, replace=False)
            row = df_copy.loc[drop_index[0]]
            df_copy = df_copy.drop(drop_index[0])  # Get rid of the example so we don't get it again this epoch

            # Get winning prototype
            closest_prototype, closest_idx = get_closest_prototype(prototypes, row)
            function = (lambda x, y: x + y)
            closest_prototype['X'] = function(closest_prototype['X'],
                                              learning_rate * (row['X'] - closest_prototype['X']))
            closest_prototype['Y'] = function(closest_prototype['Y'],
                                              learning_rate * (row['Y'] - closest_prototype['Y']))
            prototype_trace.iloc[closest_idx]['trace_X'].append(closest_prototype['X'])
            prototype_trace.iloc[closest_idx]['trace_Y'].append(closest_prototype['Y'])

        # Calculate quantization error for each epoch.
        hvq = calc_quantization_error(df, prototypes)

        # Update the minimum state if needed.
        if hvq < min_error:
            min_error = hvq
            min_error_prototypes = prototypes.copy()
            min_error_trace = prototype_trace.copy()
        quantization_errors[amt_protos - 1].append(hvq)

    # store values for this amount of prototypes
    final_hvq.append(hvq)
    final_protos.append(prototypes)
    final_proto_trace.append(prototype_trace)
    amt_protos += 1

# Plot learning curve
print("\nMinimum quantization error reached: " + str(min(quantization_errors)))
i = 1
for protos, trace in zip(final_protos, final_proto_trace):
    plot_data_points(df, protos, trace, title=f"Final Prototype Positions with {i} Prototypes")
    i += 1
    plot_curve(quantization_errors[i - 1], title=f"Learning Curve after {len(quantization_errors)} epochs",
               x_lab="Epoch", y_lab="Quantization Error")

# plot final Hvq as function of K
plot_curve(final_hvq, title="Final $H_{vq}$ values for different K values", x_lab="K", y_lab="Final Quantization Error",
           ticks=range(1, len(final_hvq) + 1))
plt.show()
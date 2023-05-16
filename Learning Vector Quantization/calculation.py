import math
import random
import numpy as np
import pandas as pd


def calc_distance(row, prototype):
    return math.sqrt((row['X']-prototype['X'])**2 + (row['Y']-prototype['Y'])**2)


def get_closest_prototype(prototypes, row):
    smallest_dist = float('inf')
    for index, prototype in prototypes.iterrows():
        curr_dist = calc_distance(row, prototype)
        if curr_dist < smallest_dist:
            smallest_dist = curr_dist
            closest_prototype = prototype
            closest_idx = index
    return closest_prototype, closest_idx


def calc_training_error(df, prototypes):
    fail = 0
    for index, row in df.iterrows():
        closest_prototype, _ = get_closest_prototype(prototypes, row)
        fail += int(closest_prototype['Label']) != int(row['Label'])
    return fail / df.shape[0]


def create_prototypes_random(amt_per_class, df):
    prototypes = pd.DataFrame(columns=['X', 'Y', 'Label'])
    prototype_trace = pd.DataFrame(columns=['trace_X', 'trace_Y'])
    for i in range(0, amt_per_class * 2):
        if i < amt_per_class:
            idx = random.randint(0, 50)
        else:
            idx = random.randint(50, 100)
        proto = df.iloc[[idx]].to_numpy()[0]
        prototypes = prototypes.append({'X': proto[0], 'Y': proto[1], 'Label': proto[2]}, ignore_index=True)
        prototype_trace = prototype_trace.append({'trace_X': [proto[0]], 'trace_Y': [proto[1]]}, ignore_index=True)
    return prototypes, prototype_trace


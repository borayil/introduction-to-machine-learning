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


def calc_squared_dist(row, winner):
    return calc_distance(row, winner)**2


def calc_quantization_error(df, prototypes):
    Hvq = 0
    # In words: For each data point in df, determine winner. 
    # Then compute squared distance between point and winner, and sum all these distances up.
    for index, row in df.iterrows():
        closest_prototype, _ = get_closest_prototype(prototypes, row)
        Hvq += calc_squared_dist(row, closest_prototype)
    return Hvq


def create_prototypes_random(amt, df):
    prototypes = pd.DataFrame(columns=['X', 'Y'])
    prototype_trace = pd.DataFrame(columns=['trace_X', 'trace_Y'])
    for i in range(0, amt):
        idx = random.randint(0, len(df.index) - 1)
        proto = df.iloc[[idx]].to_numpy()[0]
        prototypes = prototypes.append({'X': proto[0], 'Y': proto[1]}, ignore_index=True)
        prototype_trace = prototype_trace.append({'trace_X': [proto[0]], 'trace_Y': [proto[1]]}, ignore_index=True)
    return prototypes, prototype_trace


def create_prototypes_subpar(amt, df):
    max_x = df['X'].max()
    max_y = df['Y'].max()
    prototypes = pd.DataFrame(columns=['X', 'Y'])
    prototype_trace = pd.DataFrame(columns=['trace_X', 'trace_Y'])
    for i in range(0, amt):
        # chose values that increase and are definitely far away from the data
        x = max_x * 2 + 10 * i
        y = max_y * 2 + 10 * i
        prototypes = prototypes.append({'X': x, 'Y': y}, ignore_index=True)
        prototype_trace = prototype_trace.append({'trace_X': [x], 'trace_Y': [y]}, ignore_index=True)
    return prototypes, prototype_trace

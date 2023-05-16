import math
import random
import statistics
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


def calc_error(df, prototypes):
    fail = 0
    for index, row in df.iterrows():
        closest_prototype, _ = get_closest_prototype(prototypes, row)
        fail += int(closest_prototype['Label']) != int(row['Label'])
    return fail / df.shape[0]


def create_prototypes_random(amt_per_class, df):
    prototypes = pd.DataFrame(columns=['X', 'Y', 'Label'])
    prototype_trace = pd.DataFrame(columns=['trace_X', 'trace_Y'])
    class_zero_df = df.loc[df['Label'] == 0].reset_index(drop=True)
    class_one_df = df.loc[df['Label'] == 1].reset_index(drop=True)
    # create list of possible indexes to ensure different prototypes are chosen
    zero_idx_list = list(range(class_zero_df.shape[0]))
    one_idx_list = list(range(class_one_df.shape[0]))
    for i in range(0, amt_per_class):
        idx_zero = random.choice(zero_idx_list)
        idx_one = random.choice(one_idx_list)
        zero_idx_list.remove(idx_zero)
        one_idx_list.remove(idx_one)
        proto_zero = class_zero_df.iloc[[idx_zero]].to_numpy()[0]
        proto_one = class_one_df.iloc[[idx_one]].to_numpy()[0]
        prototypes = prototypes.append({'X': proto_zero[0], 'Y': proto_zero[1], 'Label': proto_zero[2]},
                                       ignore_index=True)
        prototype_trace = prototype_trace.append({'trace_X': [proto_zero[0]], 'trace_Y': [proto_zero[1]]},
                                                 ignore_index=True)
        prototypes = prototypes.append({'X': proto_one[0], 'Y': proto_one[1], 'Label': proto_one[2]},
                                       ignore_index=True)
        prototype_trace = prototype_trace.append({'trace_X': [proto_one[0]], 'trace_Y': [proto_one[1]]},
                                                 ignore_index=True)
    return prototypes, prototype_trace


def cross_validation(list_df, amt_subsets, amt_protos, t_max, learning_rate):
    training_error_list = []
    testing_error_list = []
    prototype_list = []
    prototype_trace_list = []
    for test_set_idx in range(amt_subsets):
        print(f"Test set index = {test_set_idx}")
        # determine which set is the testing set and prepare the training set
        test_set = list_df[test_set_idx].reset_index(drop=True)
        training_set = pd.concat(list_df[:test_set_idx] + list_df[test_set_idx + 1:], ignore_index=True)
        prototypes, prototype_trace = create_prototypes_random(amt_protos, training_set)
        # perform the current training
        train_err, protos, trace = epoch_loop(t_max, prototypes, learning_rate, prototype_trace, training_set)
        # store values
        training_error_list.append(train_err)
        testing_error_list.append(calc_error(test_set, protos))
        prototype_list.append(protos)
        prototype_trace_list.append(trace)
    return (statistics.mean(training_error_list), statistics.stdev(training_error_list)),\
           (statistics.mean(testing_error_list), statistics.stdev(testing_error_list)),\
           prototype_list, prototype_trace_list


def epoch_loop(t_max, prototypes, learning_rate, prototype_trace, training_set):
    curr_errors = []
    print("Training...")
    for i in range(0, t_max):
        for index, row in training_set.iterrows():
            # randomize
            training_set = training_set.sample(frac=1).reset_index(drop=True)
            # for current point check closest prototype
            closest_prototype, closest_idx = get_closest_prototype(prototypes, row)
            # update the prototype
            function = (lambda x, y: x + y) if int(closest_prototype['Label']) == int(row['Label']) else (
                lambda x, y: x - y)
            closest_prototype['X'] = function(closest_prototype['X'],
                                              learning_rate * (row['X'] - closest_prototype['X']))
            closest_prototype['Y'] = function(closest_prototype['Y'],
                                              learning_rate * (row['Y'] - closest_prototype['Y']))
            prototype_trace.iloc[closest_idx]['trace_X'].append(closest_prototype['X'])
            prototype_trace.iloc[closest_idx]['trace_Y'].append(closest_prototype['Y'])
        curr_error = calc_error(training_set, prototypes)
        curr_errors.append(curr_error)
    print("Training complete!", end="\n\n")
    return curr_error, prototypes, prototype_trace

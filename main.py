import sys
import pandas as pd
import scipy.stats as stats
import scipy.io as sio
import numpy as np
from node import TreeNode
# TODO: extract each thing in capitals in different files

# Macros
CLEAN_DATA_PATH = 'Data/cleandata_students.mat'
AU_INDICES = list(range(1, 46))
emotion = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}

# Loading data from mat files
def load_raw_data():
    print("Loading raw data...")
    mat_contents = sio.loadmat(CLEAN_DATA_PATH)
    data = mat_contents['x']   # entries/lines which contain the activated AU/muscles
    labels = mat_contents['y'] # the labels from 1-6 of emotions for each entry in data
    print("Raw data loaded...")
    return labels, data

# Converting data to DataFrame format
def to_dataframe(labels, data):
    print("Converting to data frame started...")
    df_labels = pd.DataFrame(labels)
    df_data = pd.DataFrame(data, columns=AU_INDICES)
    print("Converting to data frame done...")
    return df_labels, df_data

# Filter a vector in df format to be 1 where we have
# this certain emotion and 0 otherwise
# emotion is an int
def filter_for_emotion(df, emotion):
    print("Filtering to binary targets for emotion... ")
    emo_df = [] * 45
    emo_df = np.where(df == emotion, 1, 0)
    print("Filtering done...")
    return pd.DataFrame(emo_df)

# Decision tree learning
# Binary data   - binary matrix with N rows and 45 cols
#               - each row is a list of AUs that describe
#               - a certain emotion
# Attributes    - the list of Action Units (AU) that are candidates
#               - for the best attribute at a certain point
# Target vector - emotions vector with 1 for a certain emotion
#               - and 0 otherwise
def decision_tree(examples, attributes, bin_targets):
    if examples.empty or not attributes or bin_targets.empty:
        return None

    all_same = check_all_same(bin_targets)

    if all_same:
        return TreeNode(None, True, bin_targets.iloc[0].iloc[0])
    elif not attributes:
        # Majority Value
        return TreeNode(None, True, majority_value(bin_targets))
    else:
        best_attribute = choose_best_decision_attr(examples, attributes, bin_targets)
        tree = TreeNode(best_attribute)
        for vi in range(0, 2):
            examples_i = examples.loc[examples[best_attribute] == vi]
            indices = examples_i.index.values
            bin_targets_i = bin_targets.ix[indices]

            if examples_i.empty:
                # Majority Value
                return TreeNode(None, True, majority_value(bin_targets))
            else:
                attr = set(attributes)
                attr.remove(best_attribute)
                tree.set_child(vi, decision_tree(examples_i, attr, bin_targets_i))

        return tree

# Helper functions
def check_all_same(df):
    return df.apply(lambda x: len(x[-x.isnull()].unique()) == 1 , axis = 0).all()

def majority_value(bin_targets):
    res = stats.mode(bin_targets[0].values)[0][0]
    return res

def choose_best_decision_attr(examples, attributes, bin_targets):
    def f(eg_val, attr_val, attribute):
        return pd_joint[(pd_joint[0] == eg_val) & (pd_joint[attribute] == attr_val)].shape[0]

    pd_joint = pd.concat([bin_targets, examples], axis=1)

    def get_gain(attr):
        p1, n1, p0, n0 = f(1,1,attr), f(0,1,attr), f(1,0,attr), f(0,0,attr)
        return gain(p1+p0, n1+n0, p0, n0, p1, n1)

    all_gains = [(get_gain(a), a) for a in attributes]
    (max_gain, index_gain) = max(all_gains, key=lambda x: x[0]) if all_gains else (-1, -1)
    return index_gain


# Gain(attribute) = I(p, n) – Remainder(attribute)
def gain(p, n, p0, n0, p1, n1):
    return get_info_gain(p, n) - get_remainder(p, n, p0, n0, p1, n1)

# Information Gain I
# I(p, n) = − p+n log 2 ( p+n ) − p+n log 2 ( p+n ) and
def get_info_gain(p, n):

    if p + n == 0:
        return 0

    term = float(p / (p + n))
    return stats.entropy([term, 1 - term], base=2)

# Remainder(attribute) = (p0 + n0)/(p + n) * I(p0, n0) + (p1 + n1)/(p + n) * I(p1, n1)
def get_remainder(p, n, p0, n0, p1, n1):
    return ((p0 + n0)/(p + n)) * get_info_gain(p0, n0) + ((p1 + n1)/(p + n)) * get_info_gain(p1, n1) if p+n != 0 else 0

# Computes list [(start, end)] of the limits of K segments used in cross validation in a df of length N
def preprocess_for_cross_validation(N, K = 10):
    seg_size = int(N/K)     # Size of a block of examples/targets
    segments = [(i - seg_size, i - 1) for i in range(seg_size, N-+1, seg_size)]
    segments[-1] = (segments[-1][0], N-1)
    return segments

# Slices the initial dataframes (df_data, binary targets) into dataframes for training and for testing
def get_train_test_segs(test_seg, N, slice_func):
    (test_start, test_end) = test_seg
    test_df_data, test_df_targets = slice_func(test_start, test_end)

    if test_start == 0:                                       # test is first segment
        train_df_data, train_df_targets = slice_func(test_end + 1, N-1)
    elif test_end == N - 1:                                   # test is last segment
        train_df_data, train_df_targets = slice_func(0, test_start-1)
    else:                                                     # test is middle segment
        data_p1,targets_p1 = slice_func(0, test_start-1)
        data_p2, targets_p2 = slice_func(test_end+1, N-1)

        train_df_data = pd.concat([data_p1, data_p2], axis=0)
        train_df_targets = pd.concat([targets_p1, targets_p2], axis=0)

    return test_df_data, test_df_targets, train_df_data, train_df_targets

def cross_validation_error(df_labels, N, df_data, segments):
    def slice_segments(from_index, to_index):
        return df_data[from_index : to_index + 1], binary_targets[from_index : to_index + 1] # to change to binary targets !!!!!

    error_list = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
    for e in emotion.keys():
        total_error_for_emotion = 0
        error_list[1] = 2
        print("/\ Decision tree building for emotion:", e)
        binary_targets = filter_for_emotion(df_labels, emotion[e])
        for test_seg in segments:
            test_df_data, test_df_targets, train_df_data, train_df_targets = get_train_test_segs(test_seg, N, slice_segments)
            root = decision_tree(train_df_data, set(AU_INDICES), train_df_targets)
            # root = decision_tree(df_data, set(AU_INDICES), binary_targets)
            print("/\ Decision tree built")
            # TreeNode.traverse(root)
            count = 0
            # Counts number of incorrectly predicted tests
            for i in test_df_data.index.values:
               count += 1 - TreeNode.dfs(root, test_df_data.loc[i], test_df_targets.loc[i].at[0])

            error = count / len(test_df_targets)
            total_error_for_emotion += error
            print()

        total_error_for_emotion /= 10
        error_list[e] = total_error_for_emotion
        print()
        print("Total error:", total_error_for_emotion)
        print()

# Testing
def main():

    labels, data = load_raw_data()
    df_labels, df_data = to_dataframe(labels, data)

    # Number of examples
    N = df_labels.shape[0]   
    segments = preprocess_for_cross_validation(N)
    print("----------------------------------- LOADING COMPLETED ----------------------------------- \n")
    cross_validation_error(df_labels, N, df_data, segments)


if __name__ == "__main__": main()

import pandas as pd
import scipy.stats as stats
import scipy.io as sio
import numpy as np
import os
import math
from node import TreeNode
# TODO: extract each thing in capitals in different files

# Macros
clean_data_path = 'Data/cleandata_students.mat'
au_indices = list(range(1, 46))
emotion = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}

# Loading data from mat files
def load_raw_data():
    mat_contents = sio.loadmat(clean_data_path)
    data = mat_contents['x']   # entries/lines which contain the activated AU/muscles
    labels = mat_contents['y'] # the labels from 1-6 of emotions for each entry in data
    return labels, data

# Converting data to DataFrame format
def to_dataframe(labels, data):
    df_labels = pd.DataFrame(labels)
    df_data = pd.DataFrame(data, columns=au_indices)
    return df_labels, df_data

'''
Filter a vector in df format to be 1 where we have
this certain emotion and 0 otherwise
'''
def filter_for_emotion(df, emotion):
    # emotion is an int
    emo_df = [] * 45
    emo_df = np.where(df == emotion, 1, 0)
    return pd.DataFrame(emo_df)

'''
Decision tree learning

Binary data   - binary matrix with N rows and 45 cols
              - each row is a list of AUs that describe
              - a certain emotion

Attributes    - the list of Action Units (AU) that are candidates
              - for the best attribute at a certain point

Target vector - emotions vector with 1 for a certain emotion
              - and 0 otherwise
'''
def decision_tree(examples, attributes, bin_targets):

    if examples.empty or not attributes or bin_targets.empty:
        return None

    all_same = check_all_same(bin_targets)

    if all_same:
        return TreeNode(bin_targets.iloc[0].iloc[0], True)
    elif not attributes:
        # Majority Value
        return TreeNode(majority_value(bin_targets), True)
    else:
        best_attribute = choose_best_decision_attr(examples, attributes, bin_targets)
        tree = TreeNode(best_attribute)
        for vi in range(0, 2):
            examples_i = examples.loc[examples[best_attribute] == vi]
            indices = examples_i.index.values
            bin_targets_i = bin_targets.ix[indices]

            if examples_i.empty:
                # Majority Value
                return TreeNode(majority_value(bin_targets), True)
            else:
                attr = set(attributes)
                attr.remove(best_attribute)
                tree.set_child(vi, decision_tree(examples_i, attr, bin_targets_i))

        return tree

# Helper functions
def check_all_same(df):
    return df.apply(lambda x: len(x[-x.isnull()].unique()) == 1 , axis = 0).all()

def majority_value(bin_targets):
    res = stats.mode(bin_targets[0].values)[0][0][0]
    print(res)
    return res

def choose_best_decision_attr(examples, attributes, bin_targets):
#    print("Best decision attribute...")
#    print(attributes)
    max_gain = 0
    index_gain = 0
    # p and n -> training data has p positive and n negative examples
    p = len(bin_targets.loc[bin_targets[0] == 1].index)
    n = len(bin_targets.loc[bin_targets[0] == 0].index)

#    print("Positives : ", p)
#    print("Negatives : ", n)

    for attribute in attributes:

        examples_pos = examples.loc[examples[attribute] == 1]        
        examples_neg = examples.loc[examples[attribute] == 0]

        index_pos = examples_pos.index.values
        index_neg = examples_neg.index.values

#        print(index_pos)
#        print("=======")
#        print(index_neg)

        p0 = 0 
        n0 = 0
        p1 = 0
        n1 = 0

        for index in index_pos:
            if bin_targets[0][index] == 1:
                p1 = p1 + 1
            else:
                n1 = n1 + 1    

        for index in index_neg:
            if bin_targets[0][index] == 1:
                p0 = p0 + 1
            else:
                n0 = n0 + 1    

#        print(p0)
#        print(n0)
#        print(p1)
#        print(n1)

        curr_gain = gain(p, n, p0, n0, p1, n1)
        if curr_gain > max_gain:
            index_gain = attribute
            max_gain = curr_gain
#    print("Max gain found ", max_gain)
    return index_gain


def gain(p, n, p0, n0, p1, n1):
    # Gain(attribute) = I(p, n) – Remainder(attribute)
    return get_info_gain(p, n) - get_remainder(p, n, p0, n0, p1, n1)

# Information Gain I
def get_info_gain(p, n):
    # I(p, n) = − p+n log 2 ( p+n ) − p+n log 2 ( p+n ) and
    if p + n == 0:
        return 0

    term = float(p / (p + n))
#    return -math.log(term, 2) * term - (math.log(1 - term, 2) * (1 - term))
    return stats.entropy([term, 1 - term], base=2)

# Remainder
def get_remainder(p, n, p0, n0, p1, n1):
    # Remainder(attribute) = (p0 + n0)/(p + n) * I(p0, n0) + (p1 + n1)/(p + n) * I(p1, n1)
    return (p0 + n0)/(p + n) * get_info_gain(p0, n0) + (p1 + n1)/(p + n) * get_info_gain(p1, n1)

def test(df):
    print(df)
    print("====")
    df = df.loc[df[1] == 0]
    print(df)
    print("====")
    print(df.index.values)

def main():

    # Testing
    labels, data = load_raw_data()
    df_labels, df_data = to_dataframe(labels, data)

    binary_targets = filter_for_emotion(df_labels, emotion['surprise'])

    root = decision_tree(df_data, set(au_indices), binary_targets)
    print(root)

if __name__ == "__main__": main()

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
    return pd.concat([df_labels, df_data], axis=1)

# Filter a vector in df format to be 1 where we have
# this certain emotion and 0 otherwise
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
    print(bin_targets)

    all_same = bin_targets.apply(lambda x: len(x[-x.isnull()].unique()) == 1 , axis = 0).all()

    if all_same:
        leaf = TreeNode(bin_targets[0][0])
        leaf.set_leaf()
        return leaf;
    elif not attributes:
        # Majority Value
        return TreeNode(majority_value(bin_targets))
    else:
        print("No")
        best_attribute = choose_best_decision_attr(examples, attributes, bin_targets)
        tree = TreeNode(best_attribute)
        for vi in range(0, 2):
            examples_i = examples.loc[examples[best_attribute] == vi]
            bin_targets_i = pd.DataFrame(examples_i.index.values)

            if examples_i.empty:
                # Majority Value
                leaf = TreeNode(majority_value(bin_targets))
                leaf.set_leaf()
                return leaf
            else:
                print(best_attribute)
                print(attributes)
                attr = set(attributes)
                tree.set_child(vi, decision_tree(examples_i, attr.remove(best_attribute), bin_targets_i))

        return tree


def majority_value(bin_targets):
    print("Majority value computing...")
    return stats.mode(bin_targets.values)[0][0][0]

def choose_best_decision_attr(examples, attributes, bin_targets):
    print("Best decision attribute...")
    print(attributes)
    max_gain = 0

    index_gain = 0
    # p and n -> training data has p positive and n negative examples
    p = len(bin_targets.loc[bin_targets[0] == 1].index)
    n = len(bin_targets.loc[bin_targets[0] == 0].index)

    for attribute in attributes:

        examples_pos = examples.loc[examples[attribute] == 1]        
        examples_neg = examples.loc[examples[attribute] == 0]

        index_pos = examples_pos.index.values
        index_neg = examples_neg.index.values

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

        curr_gain = gain(p, n, p0, n0, p1, n1)
        if curr_gain > max_gain:
            index_gain = attribute
            max_gain = curr_gain
    print("Max gain found ", max_gain)

    return index_gain


def gain(p, n, p0, n0, p1, n1):
    # Gain(attribute) = I(p, n) – Remainder(attribute)
    return get_info_gain(p, n) - get_remainder(p, n, p0, n0, p1, n1)

# Information Gain I
def get_info_gain(p, n):
    # I(p, n) = − p+n log 2 ( p+n ) − p+n log 2 ( p+n ) and
    term = float(p / (p + n))
#    return -math.log(term, 2) * term - (math.log(1 - term, 2) * (1 - term))
    return stats.entropy([term, 1 - term], base=2)

# Remainder
def get_remainder(p, n, p0, n0, p1, n1):
    # Remainder(attribute) = (p0 + n0)/(p + n) * I(p0, n0) + (p1 + n1)/(p + n) * I(p1, n1)
    return (p0 + n0)/(p + n) * get_info_gain(p0, n0) + (p1 + n1)/(p + n) * get_info_gain(p1, n1)

def main():

    # Testing
    labels, data = load_raw_data()
    df = to_dataframe(labels, data)
    
    root = decision_tree(df.ix[:,1:], set(au_indices), filter_for_emotion(df.ix[:,0], emotion['surprise']))# , filter_for_emotion(df, emotion['surprise']))

    print(root)

#    p = filter_for_emotion(df, emotion['surprise'])
#    print(p.ix[:,1:])


if __name__ == "__main__": main()

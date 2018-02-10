import sys
import pandas as pd
import scipy.stats as stats
import scipy.io as sio
import numpy as np
import random as rand
from node import TreeNode
import utilities as util

emotion = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}

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

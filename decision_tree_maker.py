import sys
import pandas as pd
import scipy.stats as stats
from node import TreeNode

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

'''
    Helper functions
'''
def check_all_same(df):
    return df.apply(lambda x: len(x[-x.isnull()].unique()) == 1 , axis = 0).all()

def majority_value(bin_targets):
    res = stats.mode(bin_targets[0].values)[0][0]
    return res

def choose_best_decision_attr(examples, attributes, bin_targets):
    max_gain = -sys.maxsize - 1
    index_gain = -1
    # p and n: training data has p positive and n negative examples
    p = len(bin_targets.loc[bin_targets[0] == 1].index)
    n = len(bin_targets.loc[bin_targets[0] == 0].index)

    for attribute in attributes:
        examples_pos = examples.loc[examples[attribute] == 1]
        examples_neg = examples.loc[examples[attribute] == 0]
        index_pos = examples_pos.index.values
        index_neg = examples_neg.index.values

        p0 = n0 = p1 = n1 = 0

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

    if max_gain == -sys.maxsize - 1:
        raise ValueError('Index gain is original value...')

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









# def choose_best_decision_attr(examples, attributes, bin_targets):
#     def f(eg_val, attr_val, attribute):
#         return pd_joint[(pd_joint[0] == eg_val) & (pd_joint[attribute] == attr_val)].shape[0]
#
#     pd_joint = pd.concat([bin_targets, examples], axis=1)
#
#     def get_gain(attr):
#         p1, n1, p0, n0 = f(1,1,attr), f(0,1,attr), f(1,0,attr), f(0,0,attr)
#         return gain(p1+p0, n1+n0, p0, n0, p1, n1)
#
#     all_gains = [(get_gain(a), a) for a in attributes]
#     (max_gain, index_gain) = max(all_gains, key=lambda x: x[0]) if all_gains else (-1, -1)
#     return index_gain
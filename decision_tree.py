import pandas as pd
import random as rand
import scipy.stats as stats
import sys

import utilities as util
import constants as cnst
import plot
import measures

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


    term_1 = float(p / (p + n))
    term_2 = float(n / (p + n))

    return stats.entropy([term_1, term_2], base=2)

# Remainder(attribute) = (p0 + n0)/(p + n) * I(p0, n0) + (p1 + n1)/(p + n) * I(p1, n1)
def get_remainder(p, n, p0, n0, p1, n1):
    return ((p0 + n0)/(p + n)) * get_info_gain(p0, n0) + ((p1 + n1)/(p + n)) * get_info_gain(p1, n1) if p+n != 0 else 0


'''
    predictions  - DataFrame column with predicted emotions for each test_data_df,
                 - indexes from 1 to 6
    expectations - DataFrame column wtih expected emotions, basically test_data_labels

    Computes confusion matrix by incrementing conf_matrix[expectation[i], prediction[i]]
'''
def compare_pred_expect(predictions, expectations):
    confusion_matrix = pd.DataFrame(0, index=cnst.EMOTIONS_INDICES, columns=cnst.EMOTIONS_INDICES)
    predictions, expectations = predictions.reset_index(drop=True), expectations.reset_index(drop=True)

    for index in predictions.index.values:
        e = expectations.iloc[index] - 1
        p = predictions.iloc[index] - 1
        confusion_matrix.loc[p, e] += 1

    return confusion_matrix

'''
    tree_predictions - 6 predictions from the 6 decisin trees for one emotion
    Returns index of best prediction in list, from 0 to 5.
    Uses random function for the no predictions at all or more than 2 predictions
''' 
def choose_prediction(tree_predictions):
    occurrences = [index for index, value in enumerate(tree_predictions) if value == 1]
    if len(occurrences) == 1:
        return occurrences[0]
    elif len(occurrences) == 0:
        return rand.randint(0, 5)
    else:
        return rand.choice(occurrences)

'''
    Takes your trained trees (all six) T and the features x2 and 
    produces a vector of label predictions
'''
def test_trees(T, x2):
    predictions = []
    for i in x2.index.values:
        example = x2.loc[i]
        tree_predictions = []
        for tree in T:
            prediction = TreeNode.dfs(tree, example)
            tree_predictions.append(prediction)

        prediction_choice = choose_prediction(tree_predictions)
        predictions.append(prediction_choice + 1)

    return pd.DataFrame(predictions)

def visualise(df_labels, df_data, N):
    for e in cnst.EMOTIONS_LIST:
        root = decision_tree(df_data, set(cnst.AU_INDICES), util.filter_for_emotion(df_labels, cnst.EMOTIONS_DICT[e]))
        TreeNode.plot_tree(root)

def apply_d_tree_parallel(df_labels, df_data, N):
    print(">> Running decision tree algorithm on multiple processes.\n")
    pass

'''
    Computes a confusion matrix using decison trees only.
    Does N-folds, for each of them the following algo been applied:
        - take N - 1 training data/training targets
        - make decision trees
        - gets the best prediction based on decision trees
        - compare predictions with expectations (df_test_labels)    
'''
def apply_d_tree(df_labels, df_data, N):
    print(">> Running decision tree algorithm on a single process.\n")

    def slice_segments(from_index, to_index):
        return df_data[from_index : to_index + 1], df_labels[from_index : to_index + 1]

    res = pd.DataFrame(0, index=cnst.EMOTIONS_INDICES, columns=cnst.EMOTIONS_INDICES)

    segments = util.preprocess_for_cross_validation(N)

    for test_seg in segments:
        print(">> Starting fold... from:", test_seg)
        print()

        T = []
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.get_train_test_segs(test_seg, N, slice_segments)

        for e in cnst.EMOTIONS_LIST:
            print("Building decision tree for emotion...", e)
            train_binary_targets = util.filter_for_emotion(train_df_targets, cnst.EMOTIONS_DICT[e])
            root = decision_tree(train_df_data, set(cnst.AU_INDICES), train_binary_targets)
            print("Decision tree built. Now appending...\n")
            T.append(root)
    
        print("All decision trees built.\n")
    
        predictions = test_trees(T, test_df_data)
        confusion_matrix = compare_pred_expect(predictions, test_df_targets)
        res = res.add(confusion_matrix)
    
    # res = res.div(10)
    res = res.div(res.sum(axis=1), axis=0)
    
    for e in cnst.EMOTIONS_LIST:
        print("----------------------------------- MEASUREMENTS -----------------------------------")
        print(measures.compute_binary_confusion_matrix(res, cnst.EMOTIONS_DICT[e]))

    return res
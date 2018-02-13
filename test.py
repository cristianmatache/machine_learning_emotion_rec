import pandas as pd
import numpy as np
import random as rand
import sys
from multiprocessing import Process
from multiprocessing import Queue
import time

import decision_tree as dtree
import utilities as util
import constants as cnst
import measures
import decision_forest as dforest

from node import TreeNode

def make_and_save_d_forest(df_labels, df_data, N):
    print(">> Running decision forest algorithm on multiple processes.\n")

    res = pd.DataFrame(0, index=cnst.EMOTIONS_INDICES, columns=cnst.EMOTIONS_INDICES)

    forest_T = []

    samples = dforest.split_in_random(df_data, df_labels)
    print("Building decision forest...")
    for e in cnst.EMOTIONS_LIST:
        T= []

        processes = []
        queue_list = []

        for (sample_target, sample_data) in samples:
            print("Building decision tree for emotion...", e)
            train_binary_targets = util.filter_for_emotion(sample_target, cnst.EMOTIONS_DICT[e])

            q = Queue()
            queue_list.append(q)

            process = Process(target=dtree.decision_tree_parallel, args=(sample_data, set(cnst.AU_INDICES), train_binary_targets, q))
            processes.append(process)
            process.start()

        for p in processes:
            p.join()

        for q in queue_list:
            T.append(q.get())

        forest_T.append(T)
    util.save_forest_to_file(forest_T)

def load_and_apply_d_forest(test_df_targets, test_df_data):
    forest_T = util.load_forest(len(cnst.EMOTIONS_LIST))

    predictions_forest = dforest.test_forest_trees(forest_T, test_df_data)
    confusion_matrix = dtree.compare_pred_expect(predictions_forest, test_df_targets)
    print("----------------------------------- CONFUSION MATRIX -----------------------------------\n")
    print(confusion_matrix)
    res = confusion_matrix

    diag_res = sum(pd.Series(np.diag(res),
                        index=[res.index, res.columns]))
    sum_all_res = res.values.sum()
    accuracy_res = (diag_res/sum_all_res) * 100
    print("-----------------------------------  AVERAGE ACCURACY -----------------------------------\n:", accuracy_res)

    res = res.div(res.sum(axis=1), axis=0)
    for e in cnst.EMOTIONS_LIST:
        print("----------------------------------- MEASUREMENTS -----------------------------------")
        print(measures.compute_binary_confusion_matrix(res, cnst.EMOTIONS_DICT[e]))

def make_and_save_d_tree(df_labels, df_data, N):
    print(">> Running decision tree algorithm on a single process.\n")

    res = pd.DataFrame(0, index=cnst.EMOTIONS_INDICES, columns=cnst.EMOTIONS_INDICES)

    segments = util.preprocess_for_cross_validation(N)
    T = []
    # Split data into 90% Training and 10% Testing
    validation_data, validation_targets, train_df_data, train_df_targets = util.divide_data(segments[-1], N, df_data, df_labels)


    # Train Trees
    for e in cnst.EMOTIONS_LIST:
        print("Building decision tree for emotion: ", e)
        train_binary_targets = util.filter_for_emotion(train_df_targets, cnst.EMOTIONS_DICT[e])
        root = dtree.decision_tree(train_df_data, set(cnst.AU_INDICES), train_binary_targets)
        print("Decision tree built. Now appending...")
        T.append(root)

    # Use validation data to set a priority to each tree based on which is more accurate
    percentage = []
    T_P = []
    for e in cnst.EMOTIONS_LIST:
        print("\nValidation phase for emotion: ", e)
        validation_binary_targets = util.filter_for_emotion(validation_targets, cnst.EMOTIONS_DICT[e])
        results = []
        # Calculate how accurate each tree is when predicting emotions
        for i in validation_data.index.values:
            results.append(TreeNode.dfs2(T[cnst.EMOTIONS_DICT[e]- 1], validation_data.loc[i], validation_binary_targets.loc[i].at[0]))
        ones = results.count(1)
        percentage.append(ones/len(results))
        print("Validation phase ended. Priority levels have been set.")

    print("All decision trees built.\n")

    # List containing (Tree, Percentage) tuples
    T_P = list(zip(T, percentage))
    util.save_trees_to_file(T_P)

def load_and_apply_d_trees(df_labels, df_data):
    T_P = util.load_trees(len(cnst.EMOTIONS_LIST))
    predictions = dtree.test_trees(T_P, df_data)
    confusion_matrix = dtree.compare_pred_expect(predictions, df_labels)

    print(confusion_matrix)
    # Print accuracy for each fold
    diag = sum(pd.Series(np.diag(confusion_matrix),
                        index=[confusion_matrix.index, confusion_matrix.columns]))
    sum_all = confusion_matrix.values.sum()
    accuracy = (diag/sum_all) * 100
    print("Accuracy:", accuracy)

    res = confusion_matrix


    res = res.div(res.sum(axis=1), axis=0)

    for e in cnst.EMOTIONS_LIST:
        print("----------------------------------- MEASUREMENTS -----------------------------------")
        print(measures.compute_binary_confusion_matrix(res, cnst.EMOTIONS_DICT[e]))

    return res
def convert_arguments():
    algorithm = None

    if (len(sys.argv) == 2):
        algorithm = load_and_apply_d_forest

    elif (len(sys.argv) == 3):
        tree_or_forest = sys.argv[2]
        if tree_or_forest == 'tree':
            algorithm = load_and_apply_d_trees
        elif tree_or_forest == 'forest':
            algorithm = load_and_apply_d_forest
        else:
            print("Second argument not valid")
            sys.exit()

    return algorithm

def main():

    if len(sys.argv) < 1:
        print("Please insert the name of the file you want to test on")
        sys.exit()
    if len(sys.argv) < 2:
        print("Examples are being tested on forest implementation by default")

    TEST_FILE = sys.argv[1]


    START_TIME = time.time()

    labels, data = util.load_raw_data(TEST_FILE)
    A = np.array(labels)
    labels = [row[0] for row in A]
    df_labels, df_data = util.to_dataframe(labels, data)

    algorithm = convert_arguments()

    res = algorithm(df_labels, df_data)
    print(res)
    print("----------------------------------- TOTAL EXECUTION TIME -----------------------------------\n")
    END_TIME = time.time()
    print(END_TIME - START_TIME)



if __name__ == "__main__": main()

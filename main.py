import sys
import pandas as pd
import scipy.stats as stats
import scipy.io as sio
import numpy as np
import random as rand
from node import TreeNode
import utilities as util
import decision_tree_maker as dtree
# TODO: extract each thing in capitals in different files

# Macros
AU_INDICES = list(range(1, 46))
emotion = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}


def compare_pred_expect(predictions, expectations):


    confusion_matrix = pd.DataFrame(0, index=[1, 2, 3, 4, 5, 6], columns=[1, 2, 3, 4, 5, 6])
    for index in predictions.index.values:
        e = expectations.iloc[index]
        p = predictions.iloc[index]
        confusion_matrix.loc[e, p] += 1

    return confusion_matrix

# tree_predictions - 6 predictions from the 6 DTs (1 per emotion)
def choose_prediction(tree_predictions):
    occurrences = [index for index, value in enumerate(tree_predictions) if value == 1]
    if len(occurrences) == 1:
        return occurrences[0]
    elif len(occurrences) == 0:
        return rand.randint(0, 5)
    else:
        return rand.choice(occurrences)

# Takes your trained trees (all six) T and the features x2 and produces a vector of label
# predictions. Both x2 and predictions should be in the same format as x, y provided to you.
def test_trees(T, x2, x1):
    print("Computing predictions...")
    predictions = []
    for i in x2.index.values:
        example = x2.loc[i]
        tree_predictions = []
        for tree in T:
            prediction = TreeNode.dfs(tree, example)
            tree_predictions.append(prediction)
        prediction_choice = choose_prediction(tree_predictions)
        predictions.append(prediction_choice + 1)

    print("Predictions found:", predictions)
    print("Expected:", x1)
    return pd.DataFrame(predictions)

def compute_confusion_matrix(segments, df_labels, df_data, N):
    def slice_segments(from_index, to_index):
        return df_data[from_index : to_index + 1], df_labels[from_index : to_index + 1]

    res = pd.DataFrame(0, index=[1, 2, 3, 4, 5, 6], columns=[1, 2, 3, 4, 5, 6])

    for test_seg in segments:
        print("Folding number...", test_seg[0])

        T = []
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.get_train_test_segs(test_seg, N, slice_segments)
        print(test_df_data)
        print(train_df_data)
        for e in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
            print("Building decision tree for emotion", e)
            root = dtree.decision_tree(train_df_data, set(AU_INDICES), util.filter_for_emotion(train_df_targets, emotion[e]))
            print("Decision tree built.")
            T.append(root)
        predictions = test_trees(T, test_df_data, test_df_targets)
        confusion_matrix = compare_pred_expect(predictions, test_df_targets)
        print("Confusion matrix", confusion_matrix)

        print("Confusion matrix", confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0))
#        return
        res.add(confusion_matrix)
#        CONFUSION_MATRIX_LIST.append(confusion_matrix)

#    print("Confusion matrices list", CONFUSION_MATRIX_LIST)

    return res.div(res.sum(axis=1), axis=0)


# Testing
def main():
    labels, data = util.load_raw_data()
    df_labels, df_data = util.to_dataframe(labels, data)

    # Number of examples
    N = df_labels.shape[0]
    segments = util.preprocess_for_cross_validation(N)
    print("----------------------------------- LOADING COMPLETED ----------------------------------- \n")
#    cross_validation_error(df_labels, N, df_data, segments)

    print(compute_confusion_matrix(segments, df_labels, df_data, N))

#    print(choose_prediction([0, 0, 1, 0, 0, 0]))


    # for index, row in df_labels.iterrows():
    #     print(df_labels.iloc[index])
    #     print("==============")

if __name__ == "__main__": main()

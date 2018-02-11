import pandas as pd
import random as rand
import cross_validation
import numpy as np
from node import TreeNode
import utilities as util
import decision_tree_maker as dtree

# Macros
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

    return pd.DataFrame(predictions)

def compute_confusion_matrix(segments, df_labels, df_data, N):

    def slice_segments(from_index, to_index):
        return df_data[from_index : to_index + 1], df_labels[from_index : to_index + 1]

    res = pd.DataFrame(0, index=[0, 1, 2, 3, 4, 5], columns=[0, 1, 2, 3, 4, 5])

    # print(res)

    for test_seg in segments:
        T = []
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.get_train_test_segs(test_seg, N, slice_segments)

        # CORECT
        print("Test df data")
        print(test_df_data)
        print("--------------")
        print("Test df targets")
        print(test_df_targets)
        print("==============")
        print("Train df data")
        print(train_df_data)
        print("--------------")
        print("Train df targets")
        print(train_df_targets)
        print("==============")

        for e in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
            print("Building decision tree for emotion: ", e)
            train_binary_targets = util.filter_for_emotion(train_df_targets, emotion[e])
            print("Binary train targets", train_binary_targets)
            print()

            # print("******************** TRAIN TARGETS INAINTE SI DUPA FILTRARE ************")
    #         pd_join = pd.concat([train_df_targets, train_binary_targets], axis=1)
    #         print(pd_join)

            root = dtree.decision_tree(train_df_data, set(AU_INDICES), train_binary_targets)
            print("Decision tree built.")
            T.append(root)

            # predictions = test_trees(T, test_df_data, test_df_targets)
     # confusion_matrix = compare_pred_expect(predictions, test_df_targets)
    #     # print("Confusion matrix", confusion_matrix)

    #     # print("Confusion matrix", confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0))
    #     res = res.add(confusion_matrix)


    res = res.div(res.sum(axis=1), axis=0)
    return res

AU_INDICES = list(range(1, 10))

# Testing
def main():

    MOCK_SIZE = 10

    df_data_MOCK = pd.DataFrame(np.random.randint(low=0, high=2, size=(MOCK_SIZE, MOCK_SIZE)))

    df_labels_MOCK = pd.DataFrame(np.random.randint(low=1, high=7, size=(MOCK_SIZE, 1)))

    print(df_data_MOCK)
    print("============")
    print(df_labels_MOCK)

#    labels, data = util.load_raw_data()

#    A = np.array(labels)

#    labels = [row[0] for row in A]

#    df_labels, df_data = util.to_dataframe(labels, data)

    N_MOCK = MOCK_SIZE

#    print("Df_labels", df_labels)
#    print("Df_Data", df_data)

    # Number of examples
#    N = df_labels.shape[0]
    segments = util.preprocess_for_cross_validation(N_MOCK)
    print("----------------------------------- LOADING COMPLETED ----------------------------------- \n")

    compute_confusion_matrix(segments, df_labels_MOCK, df_data_MOCK, N_MOCK)

    # for e in emotion.keys():
    #     binary_targets = util.filter_for_emotion(df_labels, emotion[e])
    #     root = dtree.decision_tree(df_data, set(AU_INDICES), binary_targets)
    #     for i in range(0, len(df_labels)):
    #         TreeNode.dfs2(root, df_data.loc[i], binary_targets.loc[i].at[0])


if __name__ == "__main__": main()

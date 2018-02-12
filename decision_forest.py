import pandas as pd
import random as rand

import decision_tree as dtree
import utilities as util
import constants as cnst

from node import TreeNode

'''
    N - number of trees in the forest
    K - number of examples (df_data) used to train each tree
'''
def split_in_random(train_df_data, train_df_targets, N = 6, K=500):
    df = pd.concat([train_df_targets, train_df_data], axis=1)
    samples = []
    for i in range(N):
        sample = df.sample(K, replace=True)
        sample_target = sample.iloc[:, :1]
        sample_data = sample.iloc[:, 1:]
        samples.append((sample_target.reset_index(drop=True), sample_data.reset_index(drop=True)))

    return samples

def choose_majority_vote(all_emotion_prediction):
    M = max(all_emotion_prediction)
    occurrences = [index for index, value in enumerate(all_emotion_prediction) if value == M]
    
    if len(occurrences) == 1:
        return occurrences[0]
    elif len(occurrences) == 0:
        return rand.randint(0, 5)
    else:
        return rand.choice(occurrences)

def test_forest_trees(forest_T, x2):
    # x2 = test_df_data
    predictions = []
    for i in x2.index.values:
        example = x2.loc[i]
        all_emotion_prediction = []
        for T in forest_T:
            emotion_prediction = []
            for tree in T: # how emotion vote
                prediction = TreeNode.dfs(tree, example)
                emotion_prediction.append(prediction)
            sum_per_emotion = sum(emotion_prediction)
            all_emotion_prediction.append(sum_per_emotion)
        print("----------------------------------- ALL EMOTION PREDICTIONS -----------------------------------\n")
        print(all_emotion_prediction)

        prediction_choice = choose_majority_vote(all_emotion_prediction)
        predictions.append(prediction_choice + 1)
    return pd.DataFrame(predictions)

'''
    Computes a confusion matrix using decison forests,
    improving the prediction accuracy.
'''
def compute_confusion_matrix_forest(df_labels, df_data, N):
    def slice_segments(from_index, to_index):
        return df_data[from_index : to_index + 1], df_labels[from_index : to_index + 1]

    res = pd.DataFrame(0, index=cnst.EMOTIONS_INDICES, columns=cnst.EMOTIONS_INDICES)

    segments = util.preprocess_for_cross_validation(N)

    for test_seg in segments:
        print(">> Starting fold... from:", test_seg)
        print()

        forest_T = []
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.get_train_test_segs(test_seg, N, slice_segments)

        samples = split_in_random(train_df_data, train_df_targets)
        print("Building decision forest...")
        for e in cnst.EMOTIONS_LIST:
            T= []
            for (sample_target, sample_data) in samples:
                print("Building decision tree for emotion...", e)
                train_binary_targets = util.filter_for_emotion(sample_target, cnst.EMOTIONS_DICT[e])
                root = dtree.decision_tree(sample_data, set(cnst.AU_INDICES), train_binary_targets)
                print("Decision tree built. Now appending...\n")
                T.append(root)
            forest_T.append(T)
        print("Forest built.\n")
        print(forest_T)

        predictions_forest = test_forest_trees(forest_T, test_df_data)
        confusion_matrix = dtree.compare_pred_expect(predictions_forest, test_df_targets)
        print("----------------------------------- CONFUSION MATRIX -----------------------------------\n")
        print(confusion_matrix)
        res = res.add(confusion_matrix)

    # res = res.div(10)
    res = res.div(res.sum(axis=1), axis=0)
    for e in cnst.EMOTIONS_LIST:
        print("----------------------------------- MEASUREMENTS -----------------------------------")
        print(measures.compute_binary_confusion_matrix(res, cnst.EMOTIONS_DICT[e]))

    return res    
import time
import pandas as pd
import random as rand
import cross_validation
import numpy as np
from node import TreeNode
import utilities as util
import decision_tree_maker as dtree
import decision_forest as forest
import measures

'''
    Macros
'''
ATTRIBUTES_NUMBER = 45
EMOTION_DICT = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
AU_INDICES = list(range(1, ATTRIBUTES_NUMBER + 1))
NUMBER_OF_EMOTIONS = 6
EMOTIONS_INDICES = [i for i in range(0, NUMBER_OF_EMOTIONS)]
EMOTIONS_LIST = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

'''
    predictions  - DataFrame column with predicted emotions for each test_data_df,
                 - indexes from 1 to 6
    expectations - DataFrame column wtih expected emotions, basically test_data_labels

    Computes confusion matrix by incrementing conf_matrix[expectation[i], prediction[i]]
'''
def compare_pred_expect(predictions, expectations):
    confusion_matrix = pd.DataFrame(0, index=EMOTIONS_INDICES, columns=EMOTIONS_INDICES)
    predictions, expectations = predictions.reset_index(drop=True), expectations.reset_index(drop=True)

    for index in predictions.index.values:
        e = expectations.iloc[index] - 1
        p = predictions.iloc[index] - 1
        confusion_matrix.loc[p, e] += 1

    return confusion_matrix

'''
    Input: List with length = 6 of tuples of the form (prediction, depth, percentage)
    prediction : Each tree's prediction for one specific example
    depth: The depth at which the perticular prediction was found on the tree
    percentage: The accuracy of that specific tree based on it's performance on the validation data

    Output: The most accurate prediction
    Three cases: 1. One tree recognized this emotion(a single "1" value in predictions)
                    => return the index of the tree
                 2. Zero trees recognized this emotion =>
                  First Criteria: Choose tree which decided to not recognize it furthest
                                    away from root (highest depth)
                  Second Criteria: Choose tree with lowest accuracy
                 3. Multiple trees recognized this emotion =>
                   First Criterion: Choose tree which recognized it closest to the root
                                    reason: more generality
                   Second Criterion: Choose tree with highest accuracy

'''
def choose_prediction(pred_depth_proc):
    predictions, depths, proc = zip(*pred_depth_proc)
    indexes = [index for index, value in enumerate(predictions) if value == 1]
    print(predictions)
    print(depths)
    print(proc)
    print(indexes)
    if len(indexes) == 1:
        return indexes[0]
    elif len(indexes) == 0:
        res = 0
        MAX = 0
        index = 0
        max_depth_indexes = []
        for i in range(0, len(depths)):
            if depths[i] > MAX:
                MAX = depths[i]
                index = i
                del max_depth_indexes[:]
                max_depth_indexes.append(i)
            elif depths[i] == MAX:
                max_depth_indexes.append(i)
        if len(max_depth_indexes) == 1:
            res = max_depth_indexes[0]
        else:
            min_proc = 100
            min_proc_index = 0
            for i in max_depth_indexes:
                if (proc[i] < min_proc):
                    min_proc = proc[i]
                    min_proc_index = i
            res = min_proc_index
        print(res)
        return res
    else:
        res = 0
        MIN = 100
        index = 0
        max_depth_indexes = []
        for i in indexes:
            if depths[i] < MIN:
                MIN = depths[i]
                index = i
                del max_depth_indexes[:]
                max_depth_indexes.append(i)
            elif depths[i] == MIN:
                max_depth_indexes.append(i)
        if len(max_depth_indexes) == 1:
            res = max_depth_indexes[0]
        else:
            max_proc = 0
            max_proc_index = 0
            for i in max_depth_indexes:
                if (proc[i] > max_proc):
                    max_proc = proc[i]
                    max_proc_index = i
            res = max_proc_index
        print(res)
        return res
'''
    Takes your trained trees (all six) T and the features x2 and
    produces a vector of label predictions
'''
def test_trees(Trees, x2):

    print("Computing predictions...")
    predictions = []
    T, proc = zip(*Trees)
    for i in x2.index.values:
        example = x2.loc[i]
        tree_predictions = []
        for i in range(0, 6):
            prediction, depth = TreeNode.dfs(T[i], example)
            tree_predictions.append([prediction, depth, proc[i]])
        prediction_choice = choose_prediction(tree_predictions)
        predictions.append(prediction_choice + 1)

    print("Predictions computed")
    return pd.DataFrame(predictions)

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
        print("*********")
        print(all_emotion_prediction)

        prediction_choice = choose_majority_vote(all_emotion_prediction)
        predictions.append(prediction_choice + 1)
    return pd.DataFrame(predictions)

'''
    Computes a confusion matrix.
    Does N-folds, for each of them the following algo been applied:
        - take N - 1 training data/training targets
        - make decision trees
        - gets the best prediction based on decision trees
        - compare predictions with expectations (df_test_labels)
'''
def compute_confusion_matrix_forest(df_labels, df_data, N):
    def slice_segments(from_index, to_index):
        return df_data[from_index : to_index + 1], df_labels[from_index : to_index + 1]

    res = pd.DataFrame(0, index=EMOTIONS_INDICES, columns=EMOTIONS_INDICES)

    segments = util.preprocess_for_cross_validation(N)

    for test_seg in segments:
        print("Starting fold from", test_seg)
        # T = []
        forest_T = []
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.get_train_test_segs(test_seg, N, slice_segments)

        samples = forest.split_in_random(train_df_data, train_df_targets)
        for e in EMOTIONS_LIST:
            T= []
            for (sample_target, sample_data) in samples:
                print("Building decision tree for emotion: ", e)
                train_binary_targets = util.filter_for_emotion(sample_target, EMOTION_DICT[e])
                root = dtree.decision_tree(sample_data, set(AU_INDICES), train_binary_targets)
                print("Decision tree built. Now appending...")
                T.append(root)
            forest_T.append(T)
        print("Forest built")
        print(forest_T)

        predictions_forest = test_forest_trees(forest_T, test_df_data)
        confusion_matrix = compare_pred_expect(predictions_forest, test_df_targets)
        print("^^^^^^^^^^^^^^^^^^^CONFUSION MATRIX^^^^^^^^^^^^^^^^^")
        print(confusion_matrix)
        res = res.add(confusion_matrix)

    #     for e in EMOTIONS_LIST:
    #         print("Building decision tree for emotion: ", e)
    #         train_binary_targets = util.filter_for_emotion(train_df_targets, EMOTION_DICT[e])
    #         root = dtree.decision_tree(train_df_data, set(AU_INDICES), train_binary_targets)
    #         print("Decision tree built. Now appending...")
    #         T.append(root)
    #
    #     print("All decision trees built")
    #
    #     predictions = test_trees(T, test_df_data)
    #     confusion_matrix = compare_pred_expect(predictions, test_df_targets)
    #     res = res.add(confusion_matrix)
    #     print("Folding ended")
    #     print()
    #
    res = res.div(10)
    for e in EMOTIONS_LIST:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(measures.compute_binary_confusion_matrix(res, EMOTION_DICT[e]))

    return res

'''
    Computes a confusion matrix.
    Does N-folds, for each of them the following algo been applied:
        - take N - 1 training data/training targets
        - make decision trees
        - gets the best prediction based on decision trees
        - compare predictions with expectations (df_test_labels)
'''
def slice_segments(from_index, to_index, df_data, df_labels):
    return df_data[from_index : to_index + 1], df_labels[from_index : to_index + 1]

def compute_confusion_matrix(df_labels, df_data, N):

    res = pd.DataFrame(0, index=EMOTIONS_INDICES, columns=EMOTIONS_INDICES)

    segments = util.preprocess_for_cross_validation(N)

    for test_seg in segments:
        print("Starting fold from", test_seg)
        T = []
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.get_train_test_segs(test_seg, N, slice_segments, df_data, df_labels)

        # CORECT
        '''
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
        '''
        K = train_df_data.shape[0]
        segs = util.preprocess_for_cross_validation(K)
        validation_data, validation_targets, train_data, train_targets = util.get_train_test_segs(segs[-1], K, slice_segments, train_df_data, train_df_targets)


        for e in EMOTIONS_LIST:
            print("Building decision tree for emotion: ", e)
            train_binary_targets = util.filter_for_emotion(train_targets, EMOTION_DICT[e])
            root = dtree.decision_tree(train_data, set(AU_INDICES), train_binary_targets)
            print("Decision tree built. Now appending...")
            T.append(root)

        prec = []
        total_accuracy = 0
        for e in EMOTIONS_LIST:
            print("VALIDATING TREE: ", e)
            validation_binary_targets = util.filter_for_emotion(validation_targets, EMOTION_DICT[e])
            results = []
            for i in validation_data.index.values:
                results.append(TreeNode.dfs2(T[EMOTION_DICT[e]- 1], validation_data.loc[i], validation_binary_targets.loc[i].at[0]))
            print("Decision tree VALIDATED, new BUILDING...")
            ones = results.count(1)
            prec.append(ones/len(results))
        Trees = list(zip(T, prec))

        print("All decision trees built")

        predictions = test_trees(Trees, test_df_data)
        confusion_matrix = compare_pred_expect(predictions, test_df_targets)
        print(confusion_matrix)
        diag = sum(pd.Series(np.diag(confusion_matrix),
                            index=[confusion_matrix.index, confusion_matrix.columns]))
        sum_all = confusion_matrix.values.sum()
        accuracy = (diag/sum_all) * 100
        total_accuracy += accuracy
        print("Accuracy:", accuracy)

        res = res.add(confusion_matrix)
        print("Folding ended")
        print()
    print("TOTAL ACCURACY: ", total_accuracy)
    res = res.div(res.sum(axis=1), axis=0)
    print(res)
    return res


# Testing
def main():
    start = time.time()

    labels, data = util.load_raw_data_clean()
    A = np.array(labels)
    labels = [row[0] for row in A]
    df_labels, df_data = util.to_dataframe(labels, data)
    # Number of examples
    N = df_labels.shape[0]
    print("----------------------------------- LOADING COMPLETED ----------------------------------- \n")

    res = compute_confusion_matrix(df_labels, df_data, N)
    print("----------------------------------- CONFUSION_MATRIX ------------------------------------ \n")

    print(res)

    end = time.time()
    print("\/\  TOTAL TIME /\/")
    print(end - start)
    # MOCK_SIZE = 5
    # df_data_MOCK = pd.DataFrame(np.random.randint(low=0, high=2, size=(MOCK_SIZE, 1)))

    # print(df_data_MOCK)
    # print(df_data_MOCK.loc[0])
    # df = df_data_MOCK.loc[df_data_MOCK[1] == 0]

    # print(df)
    # print("=========================")
    # print(df.index.values)

    # print(df_data_MOCK.ix[df.index.values])
    # print("Aici", df_data_MOCK.loc[2])
    # print("====")
    # print("Aici 2", df_data_MOCK.ix[2])


if __name__ == "__main__": main()

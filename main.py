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
from multiprocessing import Process
from multiprocessing import Queue

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

def vote_and_sum_per_emotion(T, example):
    emotion_prediction = []

    processes = []
    queue_list = []

    for tree in T: # how emotion vote
        
        q = Queue()
        queue_list.append(q)

        process = Process(target=TreeNode.dfs_queue, args=(tree, example, q))
        processes.append(process)

        process.start()

        # prediction = TreeNode.dfs(tree, example)
        # emotion_prediction.append(prediction)

    for p in processes:
        p.join()

    for q in queue_list:
        emotion_prediction.append(q.get())

    return sum(emotion_prediction)

def vote_queue(T, example, queue):
    sum_per_emotion = sum(spawn_process(T, TreeNode.dfs_queue, example))
    # sum_per_emotion = vote_and_sum_per_emotion(T, example)
    queue.put(sum_per_emotion)

def spawn_process(list_t, function, example):
    processes = []
    queue_list = []
    result = []

    for l in list_t:
        q = Queue()
        queue_list.append(q)

        process = Process(target=function, args=(l, example, q))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()

    for q in queue_list:
        result.append(q.get())    

    return result

def test_forest_trees(forest_T, x2):
    # x2 = test_df_data
    predictions = []
    for i in x2.index.values:
        example = x2.loc[i]
        all_emotion_prediction = []

        all_emotion_prediction = spawn_process(forest_T, vote_queue, example)

        # processes = []
        # queue_list = []

        # for T in forest_T:

        #     q = Queue()
        #     queue_list.append(q)

        #     process = Process(target=vote_queue, args=(T, example, q))

        #     processes.append(process)

        #     process.start()

        #     # sum_per_emotion = vote_and_sum_per_emotion(T, example)
        #     # all_emotion_prediction.append(sum_per_emotion)

        # for p in processes:
        #     p.join()

        # for q in queue_list:
        #     all_emotion_prediction.append(q.get())

        print("----------------------------------- ALL EMOTION PREDICTIONS -----------------------------------\n")
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
def compute_confusion_matrix(df_labels, df_data, N):
    def slice_segments(from_index, to_index):
        return df_data[from_index : to_index + 1], df_labels[from_index : to_index + 1]

    res = pd.DataFrame(0, index=EMOTIONS_INDICES, columns=EMOTIONS_INDICES)

    segments = util.preprocess_for_cross_validation(N)

    for test_seg in segments:
        print("Starting fold from:", test_seg)
        # T = []
        forest_T = []
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.get_train_test_segs(test_seg, N, slice_segments)

        samples = forest.split_in_random(train_df_data, train_df_targets)
        print("Building decision forest...")
        for e in EMOTIONS_LIST:
            T= []

            processes = []
            queue_list = []

            for (sample_target, sample_data) in samples:
                print("Building decision tree for emotion: ", e)
                train_binary_targets = util.filter_for_emotion(sample_target, EMOTION_DICT[e])
                # root = dtree.decision_tree(sample_data, set(AU_INDICES), train_binary_targets)
                q = Queue()
                queue_list.append(q)
                process = Process(target=dtree.decision_tree_queue, args=(sample_data, set(AU_INDICES), train_binary_targets, q))
                processes.append(process)
                process.start()
                # print("Decision tree built. Now appending...")
                # T.append(root)

            for process in processes:
                process.join()

            for q in queue_list:
                T.append(q.get())

            forest_T.append(T)
        print("Forest built.")
        print(forest_T)

        predictions_forest = test_forest_trees(forest_T, test_df_data)
        confusion_matrix = compare_pred_expect(predictions_forest, test_df_targets)
        print("----------------------------------- CONFUSION MATRIX -----------------------------------\n")
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
    # res = res.div(10)
    res = res.div(res.sum(axis=1), axis=0)
    for e in EMOTIONS_LIST:
        print("----------------------------------- MEASUREMENTS -----------------------------------")
        print(measures.compute_binary_confusion_matrix(res, EMOTION_DICT[e]))

    return res

# Testing
def main():
    START_TIME = time.time()

    labels, data = util.load_raw_data_clean()
    A = np.array(labels)
    labels = [row[0] for row in A]
    df_labels, df_data = util.to_dataframe(labels, data)

    # Number of examples
    N = df_labels.shape[0]

    print("----------------------------------- LOADING COMPLETED -----------------------------------\n")

    res = compute_confusion_matrix(df_labels, df_data, N)
    print("----------------------------------- CONFUSION_MATRIX ------------------------------------\n")
    print(res)

    END_TIME = time.time()
    print("----------------------------------- TOTAL EXECUTION TIME -----------------------------------\n")
    print(END_TIME - START_TIME)

if __name__ == "__main__": main()

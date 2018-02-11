import pandas as pd
import random as rand
import cross_validation
import numpy as np
from node import TreeNode
import utilities as util
import decision_tree_maker as dtree
import threading as thd
import queue
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
        confusion_matrix.loc[e, p] += 1

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

        thread_list = []
        queue_list = [queue.Queue()] * 6
        count = 0

        for tree in T:

            t1 = thd.Thread(target=TreeNode.dfs, args=(tree, example, queue_list[count]))

            count += 1

            t1.start()
            thread_list.append(t1)

            # prediction = TreeNode.dfs(tree, example)
            # tree_predictions.append(prediction)

        for t in thread_list:
            t.join()

        for q in queue_list:
            tree_predictions.append(q.get())


        prediction_choice = choose_prediction(tree_predictions)
        predictions.append(prediction_choice + 1)

    print("Predictions computed")
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
        print("Starting fold from", test_seg)
        T = []
        test_df_data, test_df_targets, train_df_data, train_df_targets = util.get_train_test_segs(test_seg, N, slice_segments)

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

        thread_list = []

        queue_list = []

        for e in EMOTIONS_LIST:

            print("Building decision tree for emotion: ", e)
            train_binary_targets = util.filter_for_emotion(train_df_targets, EMOTION_DICT[e])

            q = queue.Queue()
            queue_list.append(q)

            t1 = thd.Thread(target=dtree.decision_tree_queue, args=(train_df_data, set(AU_INDICES), train_binary_targets, q))

            t1.start()
            thread_list.append(t1)

            # root = dtree.decision_tree(train_df_data, set(AU_INDICES), train_binary_targets)
            # print("Decision tree built. Now appending...")
            # T.append(root)

        for t in thread_list:
            t.join()

        for q in queue_list:
            T.append(q.get())

        print("All decision trees built")

        predictions = test_trees(T, test_df_data)
        confusion_matrix = compare_pred_expect(predictions, test_df_targets)

        print(confusion_matrix)

        res = res.add(confusion_matrix)
        print("Folding ended")
        print()

    res = res.div(res.sum(axis=1), axis=0)
    return res

# Testing
def main():

    MOCK_SIZE = 10
    df_data_MOCK = pd.DataFrame(np.random.randint(low=0, high=2, size=(MOCK_SIZE, MOCK_SIZE)))
    df_labels_MOCK = pd.DataFrame(np.random.randint(low=1, high=7, size=(MOCK_SIZE, 1)))

    d = {0: [0.613636,0.095960,0.075630,0.009259,0.166667,0.009662],
         1: [0.128788,0.696970,0.033613,0.060185,0.098485,0.033816],
         2: [0.053030,0.025253,0.697479,0.013889,0.090909,0.082126],
         3: [0.053030,0.070707,0.025210,0.833333,0.098485,0.028986],
         4: [0.136364,0.050505,0.050420,0.055556,0.507576,0.043478],
         5: [0.015152,0.060606,0.117647,0.027778,0.037879,0.801932]}
    confusion_matrix_MOCK = pd.DataFrame(data=d)
    # print(confusion_matrix_MOCK)
    # print(measures.compute_binary_confusion_matrix(confusion_matrix_MOCK, 1))

    labels, data = util.load_raw_data_clean()
    A = np.array(labels)
    labels = [row[0] for row in A]
    df_labels, df_data = util.to_dataframe(labels, data)

    df_labels_MOCK = df_labels
    df_data_MOCK = df_data
    N_MOCK = MOCK_SIZE

    # Number of examples
    N = df_labels.shape[0]
    N_MOCK = N

    print("----------------------------------- LOADING COMPLETED ----------------------------------- \n")

    res = compute_confusion_matrix(df_labels_MOCK, df_data_MOCK, N_MOCK)

    print("----------------------------------- CONFUSION_MATRIX ------------------------------------ \n")
    print(res)

if __name__ == "__main__": main()

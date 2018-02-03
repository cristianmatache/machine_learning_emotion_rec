import pandas as pd
import scipy.stats as stats
import scipy.io as sio
import numpy as np
# TODO: extract each thing in capitals in different files

# MACROS
file_path1 = 'D:\Learning\Imperial\Third Year\ML-EmotionRecognition\data\Data\cleandata_students'
muscles_indices = list(range(1, 46))
emo = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
classes = {'empty': -1, 'negative': 0, 'positive': 1}

# TREE STRUCTURE - Node
class Tree:
    def __init__(self, node_label):
        self.op = node_label
        self.kids = []
        self.classification = classes['empty']

# TREE STRUCTURE - Utility functions
def flatten_tree(root):
    print(str(root.op), end='')
    if root.kids:
        for kid in root.kids:
            print('[', end='')
            flatten_tree(kid)
            print(']', end='')

def print_tree(root):
    flatten_tree(root)
    print()


# LOADING
def load_raw_data():
    mat_contents = sio.loadmat(file_path1)
    data = mat_contents['x']   # entries/lines which contain the activated AU/muscles
    labels = mat_contents['y'] # the labels from 1-6 of emotions for each entry in data
    return labels, data

def to_dataframe(labels, data):
    df_labels = pd.DataFrame(labels)
    df_data = pd.DataFrame(data, columns=muscles_indices)
    return pd.concat([df_labels, df_data], axis=1)

def filter_for_emotion(df, emotion):
    # emotion is an int
    emo_df = df.copy(deep=True)
    emo_df[0] = np.where(df[0] == emotion, 1, 0)
    return emo_df


# DECISION TREE LEARNING
def i(p, n) :
    term = p/(p+n)
    return stats.entropy([term, 1-term], base=2)

# TESTING
labels, data = load_raw_data()
df = to_dataframe(labels, data)
# print(filter_for_emotion(df, emo['surprise']))
print("-----------")
t1 = Tree(1)
t2 = Tree(2)
t3 = Tree(3)
t4 = Tree(4)
t5 = Tree(5)
t1.kids = [t2, t3]
t2.kids = [t4]
t3.kids = [t5]
print_tree(t1)
print("-----------")
# print(df)


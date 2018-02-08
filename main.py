import pandas as pd
import scipy.stats as stats
import scipy.io as sio
import numpy as np
import os
# TODO: extract each thing in capitals in different files

# Macros
clean_data_path = 'Data/cleandata_students.mat'
au_indices = list(range(1, 46))
emotion = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}

# Loading data from mat files
def load_raw_data():
    mat_contents = sio.loadmat(clean_data_path)
    data = mat_contents['x']   # entries/lines which contain the activated AU/muscles
    labels = mat_contents['y'] # the labels from 1-6 of emotions for each entry in data
    return labels, data

# Converting data to DataFrame format
def to_dataframe(labels, data):
    df_labels = pd.DataFrame(labels)
    df_data = pd.DataFrame(data, columns=au_indices)
    return pd.concat([df_labels, df_data], axis=1)

# Filder df's first column to be 1 where we have
# this certain emotion and 0 otherwise
def filter_for_emotion(df, emotion):
    # emotion is an int
    emo_df = df.copy(deep=True)
    emo_df[0] = np.where(df[0] == emotion, 1, 0)
    return emo_df

'''
Decision tree learning

Examples      - binary matrix with N rows and 45 cols
              - each row is a list of AUs that describe
              - a certain emotion

Attributes    - the list of Action Units (AU) that are candidates
              - for the best attribute at a certain point

Target vector - emotions vector with 1 for a certain emotion
              - and 0 otherwise
'''
def decision_tree(examples, attr, bin_targets):
    pass


def gain(attr):
    pass

# Information Gain I
def get_info_gain(p, n):
    term = p/(p+n)
    return stats.entropy([term, 1-term], base=2)

# Remainder
def get_remainder(attr):
    pass

def main():

    # Testing
    labels, data = load_raw_data()
    df = to_dataframe(labels, data)
    
    decision_tree(data, data[0], filter_for_emotion(df, emo['surprise']))
    print(filter_for_emotion(df, emo['surprise']))


if __name__ == "__main__": main()

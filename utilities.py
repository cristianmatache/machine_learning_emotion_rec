import pandas as pd
import scipy.stats as stats
import scipy.io as sio
import numpy as np

CLEAN_DATA_PATH = 'Data/cleandata_students.mat'
AU_INDICES = list(range(1, 46))

# Loading data from mat files
def load_raw_data():
    print("Loading raw data...")
    mat_contents = sio.loadmat(CLEAN_DATA_PATH)
    data = mat_contents['x']   # entries/lines which contain the activated AU/muscles
    labels = mat_contents['y'] # the labels from 1-6 of emotions for each entry in data
    print("Raw data loaded...")
    return labels, data

# Converting data to DataFrame format
def to_dataframe(labels, data):
    print("Converting to data frame started...")
    df_labels = pd.DataFrame(labels)
    df_data = pd.DataFrame(data, columns=AU_INDICES)
    print("Converting to data frame done...")
    return df_labels, df_data

# Filter a vector in df format to be 1 where we have
# this certain emotion and 0 otherwise
# emotion is an int
def filter_for_emotion(df, emotion):
    print("Filtering to binary targets for emotion... ")
    emo_df = [] * 45
    emo_df = np.where(df == emotion, 1, 0)
    print("Filtering done...")
    return pd.DataFrame(emo_df)

# Computes list [(start, end)] of the limits of K segments used in cross validation in a df of length N
def preprocess_for_cross_validation(N, K = 10):
    seg_size = int(N/K)     # Size of a block of examples/targets
    segments = [(i - seg_size, i - 1) for i in range(seg_size, N-+1, seg_size)]
    segments[-1] = (segments[-1][0], N-1)
    return segments

# Slices the initial dataframes (df_data, binary targets) into dataframes for training and for testing
def get_train_test_segs(test_seg, N, slice_func):
    (test_start, test_end) = test_seg
    test_df_data, test_df_targets = slice_func(test_start, test_end)

    if test_start == 0:                                       # test is first segment
        train_df_data, train_df_targets = slice_func(test_end + 1, N-1)
    elif test_end == N - 1:                                   # test is last segment
        train_df_data, train_df_targets = slice_func(0, test_start-1)
    else:                                                     # test is middle segment
        data_p1,targets_p1 = slice_func(0, test_start-1)
        data_p2, targets_p2 = slice_func(test_end+1, N-1)

        train_df_data = pd.concat([data_p1, data_p2], axis=0)
        train_df_targets = pd.concat([targets_p1, targets_p2], axis=0)

    return test_df_data, test_df_targets, train_df_data, train_df_targets
import pandas as pd

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
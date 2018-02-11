import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split

# Randomly sample 7 elements from your dataframe

# N - number of trees in the forest
# K - number of examples (df_data) used to train each tree
def split_in_random(train_df_data, train_df_targets, N = 5, K=670):
    TOTAL = train_df_targets.shape[0]

    df = pd.concat([train_df_targets, train_df_data], axis=1)
    samples = []
    for i in range(N):
        sample = df.sample(K, replace=True)
        # df = df.loc[~df.index.isin(sample.index)]
        sample_target = sample.iloc[:, :1]
        sample_data = sample.iloc[:, 1:]
        samples.append((sample_target, sample_data))

    # print(samples)
    return samples, N















d1 ={0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
     1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
     2: [1, 2, 3, 4, 5, 6, 7, 8, 9],
     3: [1, 2, 3, 4, 5, 6, 7, 8, 9],
     4: [1, 2, 3, 4, 5, 6, 7, 8, 9]}

d2 ={0: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

df1 = pd.DataFrame(data=d1)
df2 = pd.DataFrame(data=d2)

split_in_random(df1, df2)
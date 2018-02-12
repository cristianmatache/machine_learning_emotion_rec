import time
import pandas as pd
import numpy as np

import cross_validation
import utilities as util
import decision_tree as dtree
import decision_forest as dforest

import measures

from node import TreeNode


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

    res = dforest.compute_confusion_matrix_forest(df_labels, df_data, N)
    print("----------------------------------- CONFUSION_MATRIX ------------------------------------\n")
    print(res)

    END_TIME = time.time()
    print("----------------------------------- TOTAL EXECUTION TIME -----------------------------------\n")
    print(END_TIME - START_TIME)

if __name__ == "__main__": main()

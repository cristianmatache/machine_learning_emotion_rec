import unittest
import main as m
import numpy as np
import utilities as util
import decision_tree_maker as dtree
import pandas as pd
import random as rand

class TestHelpers(unittest.TestCase):

#
# (0, 1, 0, 0, 0, 1)
# (6, 8, 5, 7, 6, 8)
# (0.9468085106382979, 0.9148936170212766, 0.9468085106382979, 0.9361702127659575, 0.851063829787234, 0.925531914893617)
    def test_choose_prediction(self):
        self.assertEqual(m.choose_prediction([(1, 8, 0.94), (1, 8, 0.92)]), 0)


    # def test_choose_prediction2(self):
    #     test = [1, 0, 1, 0, 1, 0]
    #     for i in range(0, 50):
    #         self.assertTrue(test[m.choose_prediction(test)] == 1)
    #
    # def test_choose_prediction3(self):
    #     test = [0, 0, 0, 0, 0, 0]
    #     for i in range(0, 50):
    #         prediction = m.choose_prediction(test)
    #         self.assertTrue(prediction >= 0 and prediction <=5)

    def test_confusion_matrix(self):
        labels, data = util.load_raw_data_clean()
        A = np.array(labels)
        labels = [row[0] for row in A]
        df_labels, df_data = util.to_dataframe(labels, data)
        # Number of examples
        N = df_labels.shape[0]
        res = compute_confusion_matrix(df_labels_MOCK, df_data_MOCK, N_MOCK)
        pd.Series(np.diag(res), index=[res.index, res.columns])

if __name__ == "__main__":
    unittest.main()


77 74 87 74

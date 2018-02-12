import unittest
import main as m
import numpy as np
import utilities as util
import pandas as pd

class TestHelpers(unittest.TestCase):

    def test_choose_prediction(self):
        self.assertEqual(m.choose_prediction([1, 0, 0, 0, 0, 0]), 0)


    def test_choose_prediction2(self):
        test = [1, 0, 1, 0, 1, 0]
        for i in range(0, 50):
            self.assertTrue(test[m.choose_prediction(test)] == 1)

    def test_choose_prediction3(self):
        test = [0, 0, 0, 0, 0, 0]
        for i in range(0, 50):
            prediction = m.choose_prediction(test)
            self.assertTrue(prediction >= 0 and prediction <=5)

    # def test_confusion_matrix(self):
    #     labels, data = util.load_raw_data_clean()
    #     A = np.array(labels)
    #     labels = [row[0] for row in A]
    #     df_labels, df_data = util.to_dataframe(labels, data)
    #     # Number of examples
    #     N = df_labels.shape[0]
    #     res = compute_confusion_matrix(df_labels_MOCK, df_data_MOCK, N_MOCK)
    #     pd.Series(np.diag(res), index=[res.index, res.columns])

if __name__ == "__main__":
    unittest.main()

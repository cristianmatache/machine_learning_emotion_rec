import unittest
import main as m
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





if __name__ == "__main__":
    unittest.main()

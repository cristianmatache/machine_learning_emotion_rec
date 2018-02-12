import pandas as pd
import constants as cnst

def compute_binary_confusion_matrix(confusion_matrix, emotion):
    # Because confusion matrix has rows and columns indexed from 0 to 5, but emotions are from 1 to 6
    emotion -= 1
    binary_df = pd.DataFrame(0, index=[0,1], columns=[0,1])

    # Classification measures
    TP = confusion_matrix.loc[emotion, emotion]
    FP = confusion_matrix[emotion].values.sum() - TP
 
    FN = confusion_matrix.loc[emotion].values.sum() - TP
    TN = confusion_matrix.values.sum() - TP - FP - FN

    # Classificatin rate
    CR = (TP + TN) / (TP + TN + FP + FN)

    # Recall, precision rates and F1 measures
    recall1 = TP / (TP + FN)
    precision1 = TP / (TP + FP)
    F1 = 2 * precision1 * recall1 / (precision1 + recall1)

    # Recall, precision rates and F2 measures
    recall2 =  TN / (TN + FP)
    precision2 = TN / (TN + FN)
    F2 = 2 * precision2 * recall2 / (precision2 + recall2)

    UAR = (recall1 + recall2) / 2

    measures = {'CR': CR, 'UAR': UAR,
                'R1': recall1, 'P1': precision1, 'F1': F1,
                'R2': recall2, 'P2': precision2, 'F2': F2}

    binary_df.loc[0,0] = TP
    binary_df.loc[0,1] = FN
    binary_df.loc[1,0] = FP
    binary_df.loc[1,1] = TN
    # print(binary_df)

    return {cnst.EMOTIONS_LIST[emotion]: measures}

d = {0: [0.606061, 0.060606, 0.092437, 0.013889, 0.136364, 0.019324],
     1: [0.121212, 0.712121, 0.042017, 0.074074, 0.128788, 0.028986],
     2: [0.045455, 0.035354, 0.638655, 0.032407, 0.075758, 0.067633],
     3: [0.053030, 0.070707, 0.016807, 0.787037, 0.053030, 0.028986],
     4: [0.128788, 0.070707, 0.067227, 0.060185, 0.590909, 0.033816],
     5: [0.045455, 0.050505, 0.142857, 0.032407, 0.015152, 0.821256]}

confusion_matrix_MOCK = pd.DataFrame(data=d)
# print(compute_binary_confusion_matrix(confusion_matrix_MOCK, 6))

# Mock of confusion matrix
# d = {0: [0.613636, 0.095960, 0.075630, 0.009259, 0.166667, 0.009662],
#      1: [0.128788, 0.696970, 0.033613, 0.060185, 0.098485, 0.033816],
#      2: [0.053030, 0.025253, 0.697479, 0.013889, 0.090909, 0.082126],
#      3: [0.053030, 0.070707, 0.025210, 0.833333, 0.098485, 0.028986],
#      4: [0.136364, 0.050505, 0.050420, 0.055556, 0.507576, 0.043478],
#      5: [0.015152, 0.060606, 0.117647, 0.027778, 0.037879, 0.801932]}
# confusion_matrix_MOCK = pd.DataFrame(data=d)
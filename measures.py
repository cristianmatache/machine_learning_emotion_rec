import pandas as pd

EMOTIONS_LIST = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

def compute_binary_confusion_matrix(confusion_matrix, emotion):
    # Because confusion matrix has rows and columns indexed from 0 to 5, but emotions are from 1 to 6
    emotion -= 1
    binary_df = pd.DataFrame(0, index=[0,1], columns=[0,1])
    TP = confusion_matrix.loc[emotion, emotion]
    FP = confusion_matrix[emotion].values.sum() - TP
    FN = confusion_matrix.loc[emotion].values.sum() - TP
    TN = confusion_matrix.values.sum() - TP - FP - FN

    CR = (TP + TN) / (TP + TN + FP + FN)

    recall1 = TP / (TP + FN)
    precision1 = TP / (TP + FP)
    F1 = 2 * precision1 * recall1 / (precision1 + recall1)

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

    return {EMOTIONS_LIST[emotion]: measures}

# Mock of confusion matrix
# d = {0: [0.613636, 0.095960, 0.075630, 0.009259, 0.166667, 0.009662],
#      1: [0.128788, 0.696970, 0.033613, 0.060185, 0.098485, 0.033816],
#      2: [0.053030, 0.025253, 0.697479, 0.013889, 0.090909, 0.082126],
#      3: [0.053030, 0.070707, 0.025210, 0.833333, 0.098485, 0.028986],
#      4: [0.136364, 0.050505, 0.050420, 0.055556, 0.507576, 0.043478],
#      5: [0.015152, 0.060606, 0.117647, 0.027778, 0.037879, 0.801932]}
# confusion_matrix_MOCK = pd.DataFrame(data=d)

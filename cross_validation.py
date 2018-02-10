from node import TreeNode
import utilities as util
import decision_tree_maker as dtree

emotion = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
AU_INDICES = list(range(1, 46))

def cross_validation_error(df_labels, N, df_data, segments):
    def slice_segments(from_index, to_index):
        return df_data[from_index : to_index + 1], binary_targets[from_index : to_index + 1]

    error_list = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
    for e in emotion.keys():
        total_error_for_emotion = 0
        error_list[1] = 2
        print("/\ Decision tree building for emotion:", e)
        binary_targets = util.filter_for_emotion(df_labels, emotion[e])
        for test_seg in segments:
            test_df_data, test_df_targets, train_df_data, train_df_targets = util.get_train_test_segs(test_seg, N, slice_segments)
            root = dtree.decision_tree(train_df_data, set(AU_INDICES), train_df_targets)
            # root = decision_tree(df_data, set(AU_INDICES), binary_targets)
            print("/\ Decision tree built")
            # TreeNode.traverse(root)
            count = 0
            # Counts number of incorrectly predicted tests
            for i in test_df_data.index.values:
               count += 1 - TreeNode.dfs2(root, test_df_data.loc[i], test_df_targets.loc[i].at[0])

            error = count / len(test_df_targets)
            total_error_for_emotion += error
            print()

        total_error_for_emotion /= 10
        error_list[e] = total_error_for_emotion
        print()
        print("Total error:", total_error_for_emotion)
        print()

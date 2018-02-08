# TREE STRUCTURE - Node
class Node:

    def __init__(self, node_label):
        self.op = node_label
        self.left = None
        self.right = None
        self.kids = [left, right]
        self.isLeaf = false
        self.thisClass = -1

    def set_left(self, child):
        self.left = child
        kids[0] = child

    def set_right(self, child):
        self.right = child
        kids[1] = child

    def set_isLeaf(self, value):
        self.isLeaf = value

    def set_thisClass(self, value):
        self.thisClass = value

    def get_left(self, value):
        return self.left

    def get_right(self, value):
        return self.right

    def get_isEmotion():
        return isEmotion

    def get_thisClass(self):
        return thisClass


    # TREE STRUCTURE - Utility functions
    def flatten_tree(root):
        print(str(root.op), end='')
        if root.kids:
            for kid in root.kids:
                print('[', end='')
                flatten_tree(kid)
                print(']', end='')

    def print_tree(self, root):
        flatten_tree(root)
        print()

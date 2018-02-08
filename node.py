# TREE STRUCTURE - Node
class Node:

    def __init__(self, node_label):
        self.op = node_label
        self.left = None
        self.right = None
        self.isLeaf = false
        self.thisClass = -1

    def set_left(self, child):
        self.left = child

    def set_right(self, child):
        self.right = child

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
    def flatten_tree(self, root):
        print(str(root.op) + " ")
        if root.left:
            flatten_tree(self, root.left)
        if root.right:
            flatten_tree(self, root.right)

    def print_tree(self, root):
        flatten_tree(root)
        print()

# Tree Structure - TreeNode in Decision Tree
class TreeNode:

    def __init__(self, node_label):
        self.op = node_label
        self.kids = [None] * 2
        self.leaf = False

    def __str__(self):
        if self.op == None:
            return ""
        else:
            return str(self.op) + " left " + str(self.kids[0]) + " right " + str(self.kids[1])

    def set_leaf(self):
        self.leaf = True

    def set_child(self, index, child):
        self.kids[index] = child

    def get_child(self, index):
        return self.kids[index]

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

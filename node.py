#child0 -> negative
#child1 -> empty
#child2 -> positive
values = {'negative': 0, 'empty': 1, 'positive': 2}

# TREE STRUCTURE - Node
class Node:
    def __init__(self, muscle):
        self.op = node_label
        self.children = []
        self.isEmotion = false

    def add_child(self, value, child):
        if isinstance(value, str)
            self.children[values[value]] = child
        elif isinstance(value, int)
            self.children[value] = child

    def get_child(self, value):
        if type(value) is str:
            return children[values[value]]
        elif isinstance(value, int)
            return children[value]

    def get_isEmotion():
        return isEmotion

    # TREE STRUCTURE - Utility functions
    def flatten_tree(self, root):
        print(str(root.op), end='')
        if root.kids:
            for kid in root.kids:
                print('[', end='')
                flatten_tree(kid)
                print(']', end='')

    def print_tree(self, root):
        flatten_tree(root)
        print()

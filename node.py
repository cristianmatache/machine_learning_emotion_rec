import plot

# Tree Structure - TreeNode in Decision Tree
_node_index = 0
_edges = []
_labels = {}
_file_index = 0

class TreeNode:

    '''
        self.op    - a label for the corresponding node (e.g. the attribute 
                   - that the node is testing). It must be empty for the leaf node    
    
        self.kids  - a cell array which will contain the subtrees that initiate from the
                   - corresponding node.


        self.value - a label for the leaf node. Can have the following possible values:
                   - 0 - 1: the value of the examples (negative-positive, respectively) if it is the same
                            for all examples, or with value as it is defined by the MAJORITY-VALUE
                            function (in the case attributes is empty)
                   - It must bye empty for an internal node, since the tree returns a label only in the
                   - leaf node.

    '''
    def __init__(self, node_label, leaf=False, value=None):
        self.op = node_label
        self.kids = [None] * 2
        self.leaf = leaf
        self.value = value
        global _node_index
        self.index = _node_index
        _node_index += 1

    def __str__(self):
        if self.leaf:
            return str(self.value)
        return str(self.op)

    def preorder_traversal(self):
        if self.op == None:
            if self.leaf:
                return self.value
            else:
                return "null"
        else:
            left = ""

            if self.kids[0] == None:
                left = "null"
            else:
                left = self.kids[0].preorder_traversal()

            if self.kids[1] == None:
                right = "null"
            else:
                right = self.kids[1].preorder_traversal()
            return str(self.op) + ", " + left + ", " + right

    def set_child(self, index, child):
        self.kids[index] = child

    def get_child(self, index):
        return self.kids[index]

    @staticmethod
    def dfs2(root, example, expectation):
        if root.leaf:
            is_correct = root.value == expectation
            print("1" if is_correct else "0", end="")
            return 1 if is_correct else 0
        else:
            index = root.op
            if example.ix[index] == 0:
                return TreeNode.dfs2(root.kids[0], example, expectation)
            else:
                return TreeNode.dfs2(root.kids[1], example, expectation)

    @staticmethod
    def dfs(root, example):
        if root.leaf:
            return root.value
        else:
            index = root.op
            if example.loc[index] == 0:
                return TreeNode.dfs(root.kids[0], example)
            else:
                return TreeNode.dfs(root.kids[1], example)

    @staticmethod
    def _dfs_pure(root):
        global _edges
        if root.leaf:
            _labels[root.index] = root.value
        else:
            _labels[root.index] = root.op
            for kid in root.kids:
                _edges.append((root.index, kid.index))
                TreeNode._dfs_pure(kid)

    @staticmethod
    def plot_tree(root, emotion = "default_emotion"):
        global _file_index,_edges, _node_index, _labels
        _labels, _edges, _node_index = {}, [], 0
        TreeNode._dfs_pure(root)
        _file_index += 1
        plot.visualize_tree(_edges, _file_index, emotion=emotion, labels=_labels)



    @staticmethod
    def traverse(root):
        current_level = [root]
        while current_level:
            print(' '.join(str(node) for node in current_level))
            next_level = list()
            for n in current_level:

                if n.op == "'#'":
                    continue

                if n.kids[0]:
                    next_level.append(n.kids[0])
                else:
                    next_level.append(TreeNode("'#'"))
                if n.kids[1]:
                    next_level.append(n.kids[1])
                else:
                    next_level.append(TreeNode("'#'"))
            current_level = next_level

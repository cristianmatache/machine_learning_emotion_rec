# Tree Structure - TreeNode in Decision Tree
class TreeNode:
    def __init__(self, node_label, leaf=False, value=None):
        self.op = node_label
        self.kids = [None] * 2
        self.leaf = leaf
        self.value = value

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
    def dfs(root, attributes_lists, expectation):
        if root.leaf:
            is_correct = root.value == expectation
            print("1" if is_correct else "0", end="")
            return 1 if is_correct else 0
        else:
            index = root.op
            if attributes_lists.ix[index] == 0:
                return TreeNode.dfs(root.kids[0], attributes_lists, expectation)
            else:
                return TreeNode.dfs(root.kids[1], attributes_lists, expectation)

    @staticmethod
    def traverse(root):
        current_level = [root]
        while current_level:
            print(' '.join(str(node) for node in current_level))
            next_level = list()
            for n in current_level:

                if n.op == "'#'":
                    continue;

                if n.kids[0]:
                    next_level.append(n.kids[0])
                else:
                    next_level.append(TreeNode("'#'"))
                if n.kids[1]:
                    next_level.append(n.kids[1])
                else:
                    next_level.append(TreeNode("'#'"))
            current_level = next_level

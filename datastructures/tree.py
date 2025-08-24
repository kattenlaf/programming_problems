from collections import deque

class Node:
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None

# Basic Tree in which we try to maintain balance adding nodes by level
class Tree:
    def __init__(self, value):
        self.root = Node(value)

    def inorder(self):
        # Visit left most subtree, then visit the node, then visit the right subtree
        sol = []
        stack = []
        current = self.root
        while current is not None or len(stack) > 0:
            while current is not None:
                stack.append(current)
                current = current.left

            current = stack.pop()
            sol.append(current.value)
            current = current.right

        return sol

    def preorder(self):
        sol = []
        if self.root is None:
            return sol
        stack = [self.root]
        while stack:
            current = stack.pop()
            if current:
                sol.append(current.value)
                if current.right:
                    stack.append(current.right)
                if current.left:
                    stack.append(current.left)

        return sol

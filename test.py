#python
#test file
from mutility import *

class Queue():
    """Queue implemented with a linked list."""
    def __init__(self):
        self.ir = LinkedList()
    
    def enqueue(self, el):
        node = ListNode(el)
        self.ir.insert(node)

    def dequeue(self):
        result = self.ir.tail()
        if result == self.ir.nil:
            raise UnderflowError('Cannot dequeue empty queue')

        self.ir.delete(self.ir.tail())
        return result.key

    def __str__(self):
        return str(self.ir)[::-1]

def recursive_print(root):
    """ Recursively prints the nodes of a binary tree."""

    if root != None:
        print(root.key)
        recursive_print(root.left)
        recursive_print(root.right)
    
def iterative_print(root):
    """Iteratively prints the keys of a tree."""
    stack = Stack(root)

    while stack:
        top = stack.pop()
        if top != None:
            print(top.key)
            stack.push(top.right)
            stack.push(top.left)

def constant_memory_print(root):
    """ Prints a binary tree (with parent) iteratively and with constant memory.
    Args:
        root: The root of a binary tree.
        
    NOTE Discussion:
        The idea is the following: keep track of the 'current' node, and the 'previous' node. The 'current' node is the 
        node being analyzed. The 'previous' node is the node that was just previously visited. There are 3 cases:
            1a: 'Previous' is None, in which case you are at root, and have just started
            1b: 'Previous' is cur.parent, in which case you are seeing cur for the first time.
            2: 'Previous' is cur.left, in which case you have completed the left tree.
            3: 'Previous is cur.right, in which case you have finished the cur subtree and should walk back up the tree.
        """
    prev = None
    cur = root
    while cur != None:
        if prev == cur.parent:
            #Visiting cur for the first time.
            print(cur.key)
            
            #do leaf detection since it won't work otherwise.
            if cur.isleaf():
                cur, prev = prev, cur
            elif cur.left == None:
                prev = cur
                cur = cur.right
            else:
                prev = cur
                cur = cur.left
        elif prev == cur.left:
            #Completed left subtree
            #do leaf detection since it won't work otherwise.
            if cur.right == None:
                prev = cur
                cur = cur.parent
            else:
                prev = cur
                cur = cur.right

        elif prev == cur.right:
            prev = cur
            cur = cur.parent
        else:
            print(f'Program failed at node {cur.key}. prev was {prev.key}')
            return


class TreeNode():
    """Container class for tree node."""
    def __init__(self, key, left = None, right = None):
        """Creates a tree node from arguments given."""
        self.key = key
        self.left = left
        self.right = right

class BST():
    """Implementation of a bst"""
    def __init__(self):
        """Creates an empty BST"""
        self.root = None
    
    def insert(self, key):
        """Inserts an element into the BST.
        
        key must support <, >, =, <=, >="""
        if self.root == None:
            self.root = TreeNode(key, None, None)
            return

        cur = self.root
        while cur:
            prev = cur
            if cur.key < key:
                cur = cur.right
            elif cur.key > key:
                cur = cur.left
            else:
                raise ValueError('[INVALID INSERTION] non-unique key')

        if prev.key < key:
            prev.right = TreeNode(key)
        elif prev.key > key:
            prev.left = TreeNode(key)
        return

    def delete(self, key):
        """Deletes element from tree if found.
        
        key must support <, >, =, <=, >="""
        def _delete(node,key):
            if node == None:
                return None

            if node.key < key:
                node.right = _delete(node.right, key)
            elif node.key > key:
                node.left = _delete(node.left, key)
            elif node.key == key:
                #4 cases:
                #no children
                if node.left == None and node.right == None:
                    return None
                elif node.left == None and node.right != None:
                    return node.right
                elif node.left != None and node.right == None:
                    return node.left
                else:
                    #2 children
                    min = BST.min(node.right)
                    node.key = min.key
                    node.right = _delete(node.right, min.key)
            
            return node
        _delete(self.root, key)

    def list(self, key1, key2):
        """Returns a list of all elements in the tree between the keys given."""
        def _list(node, key1, key2, result):
            if node == None:
                return
            if node.key >= key1:
                _list(node.left, key1, key2, result)
            if key1 <= node.key and node.key <= key2:
                result.append(node.key)
            if node.key <= key2:
                _list(node.right, key1, key2, result)
        result = []
        _list(self.root, key1, key2, result)
        return result
    
    def _verify(self):
        #verifies that the tree satisfies the bst invariant.
        def __verify(node, l=float('-inf'), h=float('inf')):
            if node == None:
                return True
            if l > node.key or node.key > h:
                return False

            return __verify(node.left, l, node.key) and __verify(node.right, node.key, h)
        
        return __verify(self.root)

    def __str__(self):
        """Creates a human readable string from the tree."""
        result = ''
        def _h(node, depth):
            nonlocal result
            if node == None:
                return
            _h(node.right, depth + 1)
            result += '  ' * depth + f":{node.key}\n"
            _h(node.left, depth + 1)
        _h(self.root, 0)
        return result[:-1]

    @staticmethod
    def min(node):
        """Returns the smallest node of the BST rooted at node.
        
        Assumes node is non-empty."""
        if node.left == None:
            return node
        return BST.min(node.left)

    @staticmethod
    def _left_rotate(node):
        #performs a left rotate on a node, returns the new root.
        x = node
        y = node.right

        x.right = y.left
        y.left = x

        return y

    @staticmethod
    def _right_rotate(node):
        #performs a right rotate on a node, returns the new root.
        y = node
        x = node.left

        y.left = x.right
        x.right = y

        return x

class AVLTree(BST):
    """Implementation of an AVL Tree, a type of self balancing BST."""
    def insert(self, key):
        def _insert(node, key):
            if node == None:
                leaf = TreeNode(key)
                leaf.height = 0
                return leaf

            if node.key < key:
                node.right = _insert(node.right, key)
            if node.key >= key:
                node.left = _insert(node.left, key)

            #rebalance node
            return AVLTree._rebalance(node)
        self.root = _insert(self.root, key)

    def delete(self, key):
        def _delete(node, key):
            if node == None:
                return None
            
            if node.key < key:
                #right subtree recurse
                node.right = _delete(node.right, key)
                return node
            elif node.key >= key:
                #left subtree recurse
                node.left = _delete(node.left, key)
                return node
            
            #node matches key
            #4 cases:
            if node.left == None and node.right == None:
                return None
            elif node.left == None and node.right != None:
                return node.right
            elif node.left != None and node.right == None:
                return node.left
            else:
                min = BST.min(node.right)
                node.key = min.key
                node.right = _delete(node.right, node.key)

            #in any case, rebalance the tree if necessary, then return root.
            return AVLTree._rebalance(node)
            
        self.root = _delete(self.root, key)
    
    def _verify(self):
        if not BST._verify(self):
            return False
        #verifies if the tree satisfies height property

        stack = [self.root]
        while stack:
            node = stack.pop()
            if node != None:
                if abs(AVLTree.weight(node)) > 1:
                    return False
                stack.append(node.right)
                stack.append(node.left)
        return True

    @staticmethod
    def height(node):
        if node == None:
            return -1
        else:
            return node.height

    @staticmethod
    def weight(node):
        if node == None:
            return -1
        #returns the weight of the node
        left_height = AVLTree.height(node.left)
        right_height = AVLTree.height(node.right)
        return right_height - left_height

    @staticmethod
    def _left_rotate(node):
        x = node
        y = node.right

        node = BST._left_rotate(node)
        x.height = max(AVLTree.height(x.left),
                       AVLTree.height(x.right)) + 1
        y.height = max(AVLTree.height(y.left),
                       AVLTree.height(y.right)) + 1
        return node
    @staticmethod
    def _right_rotate(node):
        y = node
        x = node.left

        node = BST._right_rotate(node)
        y.height = max(AVLTree.height(y.left),
                       AVLTree.height(y.right)) + 1
        x.height = max(AVLTree.height(x.left),
                       AVLTree.height(x.right)) + 1
        
        return node
    
    @staticmethod
    def _rebalance(node):
        #rebalances tree rooted at node, and returns it.
        weight = AVLTree.weight(node)

        if weight == 2:
            #weighted right
            x = node
            y = node.right

            if AVLTree.weight(y) == -1:
                y = AVLTree._right_rotate(y)
            return AVLTree._left_rotate(x)
        elif weight == -2:
            #weightedl left:
            y = node
            x = node.left

            if AVLTree.weight(x) == 1:
                y.left = AVLTree._left_rotate(x)
            return AVLTree._right_rotate(y)
        else:
            #valid weight
            node.height = max(AVLTree.height(node.left,),
                                AVLTree.height(node.right)) + 1
            return node

if __name__ == '__main__':
    pass
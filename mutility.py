#My Basic utility (mutility) functions

from datetime import datetime
from collections import deque

import math
import sys


#Useful error definition
class Heap:
    def __init__(self, key=lambda x,y : x < y):
        self.repr = []
        self.key = key #lambda x,y: returns true when x < y
    
    def __str__(self):
        return str(self.repr)

    def getMin(self, alt=None):
        if len(self.repr) == 0:
            return alt
        return self.repr[0]
    
    def popMin(self):
        if len(self.repr) == 1:
            return self.repr.pop()

        m = self.repr[0]
        self.repr[0] = self.repr.pop()
        self._heapify(0)
        return m
    
    def insert(self, el):
        self.repr.append(el)
        i = len(self.repr) - 1
        while i > 0 and self.key(self.repr[i], self.repr[(i - 1) // 2]):
            self.repr[(i - 1) // 2], self.repr[i] = self.repr[i], self.repr[(i - 1) // 2]
            i = (i - 1) // 2

            
    def _heapify(self, i):
        left = 2 * i + 1
        right = 2 * i + 2
        
        smallest = i
        if left < len(self.repr) and self.key(self.repr[left], self.repr[i]):
            smallest = left
        if right < len(self.repr) and self.key(self.repr[right], self.repr[smallest]):
            smallest = right
        
        if smallest != i:
            self.repr[i], self.repr[smallest] = self.repr[smallest] , self.repr[i]
            self._heapify(smallest)

    def __len__(self):
        return len(self.repr)

class UnderflowError(Exception):
    """Error class for underflow exception."""
    pass

#Useful object definitions
class ListNode():
    def __init__(self, key=None, next=None, prev=None):
        self.key = key
        self.next = next
        self.prev = prev

class LinkedList():
    """A linked list implementation"""
    def __init__(self, *elements):
        """Instatiates a linked list.
        Args:
            elements(optional): an iterable object specifying the initial objects in the linked list. (inserts in reverse order)"""
        self.nil = ListNode()
        self.nil.next = self.nil
        self.nil.prev = self.nil

        for el in elements:
            self.insert(ListNode(el))
    
    def head(self):
        return self.nil.next
    def tail(self):
        return self.nil.prev

    def insert(self, x):
        """Inserts node x into the front of the list."""
        x.next = self.nil.next
        x.prev = self.nil
        self.nil.next.prev = x
        self.nil.next = x
        
        pass

    def delete(self, obj):
        """Deletes objects from list."""
        obj.prev.next = obj.next
        obj.next.prev = obj.prev
        pass

    def search(self, key):
        h = self.nil.next
        while h != self.nil:
            if h.key == key:
                return h
            h = h.next
        return self.nil
    
    def reverse(self):
        """Reverses the list in place."""
        cur = self.head()
        self.nil.next, self.nil.prev = self.nil.prev, self.nil.next

        while cur != self.nil:
            cur.next, cur.prev = cur.prev, cur.next
            cur = cur.prev
        return

    def __str__(self):
        head = self.nil.next
        output = ''
        while head != self.nil:
            output += f"{head.key},"
            head = head.next
        return output[0:-1]

    @staticmethod
    def union(l1, l2):
        """Returns the union of 2 doubly linked lists, with l1 as the head. Deletes l2."""
        #l1 is the new head.
        l1_tail = l1.tail()
        l2_head = l2.head()
        l2_tail = l2.tail()

        if l2_head != l2.nil:
            l1_tail.next = l2_head
            l2_head.prev = l1_tail
            l2_tail.next = l1.nil

        del l2.nil
        del l2
        return l1

class TreeNode():
    """Container class for tree node."""
    def __init__(self, key, left = None, right = None, height=0):
        """Creates a tree node from arguments given."""
        self.key = key
        self.left = left
        self.right = right
        self.height = 0

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
                cur = cur.left

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
        lca = self._lca(key1, key2)
        result = []
        
        def _node_list(node, l, h, result):
            if node == None:
                return
            if l<=node.key and node.key <= h:
                result.append(node.key)
            if l <= node.key:
                _node_list(node.left, l, h, result)
            if h >= node.key:
                _node_list(node.right, l, h, result)

        _node_list(lca, key1, key2, result)
        return result
    
    def successor(self, key):
        #finds the successor to key
        #Args:
        #   key: A key
        #Returns successor key if it exists, or None otherwise
        #
        #Raises:
        #   keyerror: an error if key is not in tree

        path = []
        cur = self.root
        while cur and cur.key != key:
            path.append(cur)
            if cur.key >= key:
                #look left
                cur = cur.left
            else:
                cur = cur.right
        
        if cur == None:
            raise KeyError
        if cur.right != None:
            return BST.min(cur.right).key
        
        #backtrack up tree looking for first parent of type <greater>
        i = len(path) - 1
        while i >= 0 and path[i].key < cur.key:
            cur = path[i]
            i -= 1

        if i < 0:
            return None
        return path[i].key


    def predecessor(self, key):
        path = []
        cur = self.root
        while cur and cur.key != key:
            path.append(cur)
            if cur.key >= key:
                #look left
                cur = cur.left
            else:
                cur = cur.right
        
        if cur == None:
            raise KeyError
        if cur.left != None:
            return BST.max(cur.left).key
        
        #backtrack up tree looking for first parent of type <greater>
        i = len(path) - 1
        while i >= 0 and path[i].key > cur.key:
            cur = path[i]
            i -= 1

        if i < 0:
            return None
        return path[i].key


    def _lca(self, key1, key2):
        #Finds the lowest common ancestor to key1 and key2.
        node = self.root
        
        while node and not( key1 <= node.key <= key2):
            if node.key > key2:
                node = node.left
            else:
                node = node.right
        
        return node
        

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
        if not node:
            return None
        if node.left == None:
            return node
        return BST.min(node.left)

    @staticmethod
    def max(node):
        """Returns the greatest node of the BST rooted at node.
        
        Assumes node is non-empty."""
        if not node:
            return None
        if node.right == None:
            return node
        return BST.max(node.right)

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
                node = TreeNode(key)
            elif node.key < key:
                node.right = _insert(node.right, key)
            elif node.key >= key:
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
            elif node.key > key:
                #left subtree recurse
                node.left = _delete(node.left, key)
            else:
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
        #verifies if the tree satisfies height property
        if not BST._verify(self):
            return False

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
        node.height = AVLTree.height(node)
        weight = AVLTree.weight(node)

        if weight == 2:
            #weighted right
            x = node
            y = node.right

            if AVLTree.weight(y) == -1:
                x.right = AVLTree._right_rotate(y)
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

class Stack():
    """A stack is an dynamic abstract data type supporting:
        Push, which adds an element to the collection, and
        Pop, which removes the most recently added element that was not yet removed.
        """
    def __init__(self, *elements):
        """Creates an empty instance of a stack."""
        self.stack = [el for el in elements]
    def pop(self):
        """Pops the top element off the stack."""
        if len(self.stack) == 0:
            raise IndexError('Cannot pop an empty stack.')
        return self.stack.pop()

    def push(self, el):
        """Pushes element onto the stack."""
        self.stack.append(el)
        return

    def __str__(self):
        return str(self.stack)

    def __bool__(self):
        return len(self.stack) != 0

class Queue():
    """A Queue is a dynamic collection of elements supporting the following operations:
        Enqueue: Places a new item into the queue.
        Dequeu: Removes the earliest item placed into the queue."""
    def __init__(self, length):
        """Creates an empty instance of a queue.
        
        Args: 
            length: An argument specifying the length of the queue.
        """
        self.queue = [None for _ in range(length)]
        self.head = 0
        self.tail = 0
        self.length = length

    def is_full(self):
        if self.head == self.tail and self.queue[self.head] != None:
            return True
        return False

    def is_empty(self):
        return self.head == self.tail

    def enqueue(self, el):
        """Places an item into the queue.
        Args:
            el: the element to be placed.
        Raises:
            OverflowError: an exception if the queue is already full."""
        if self.is_full():
            raise OverflowError('Queue is full')
        self.queue[self.head] = el
        self.head = (self.head + 1) % self.length
    
    def dequeue(self):
        """Removes the earliest item from the queue.
        Raises:
            UnderflowError: An exception if there aren't enough items in the queue."""
        
        if self.is_empty():
            raise UnderflowError('Queue is empty.')
        
        self.queue[self.tail] = None
        self.tail = (self.tail + 1) % self.length

    def __str__(self):
        result = []
        c = 0
        while c < self.length and self.queue[(c + self.tail) % self.length] != None:
            result.append(self.queue[(c + self.tail) % self.length])
        return str(result) 

class Node():
    """Vertex node in a graph"""
    def __init__(self, key):
        self.key = key
    
    def __hash__(self):
        return hash(self.key)

    def __eq__(self, x):
        return x.key == self.key

    def __str__(self):
        return str(self.key)

class Graph():
    """A graph G = (V,E), is a collection of vertices (V) and edges (E). """
    def __init__(self, vertices = [], edges = []):
        """Instatiates a graph object."""

        #each node in the graph is stored as a key:adjacency_list pair.
        self.adj = {}

        for v in vertices:
            self.insertNode(v)

        for s,d in edges:
            self.insertEdge(s,d)
    
    def insert(self, key, key2 = None):
        if key2 == None:
            self.insertNode(key)
        else:
            self.insertEdge(key, key2)

    def insertNode(self, key):
        """Inserts a new node into the graph.
        Args:
            key: key of the new node to be added
        Raises:
            KeyError: An exception if the key is already being used.
        """
        if self.adj.get(key) != None:
            raise KeyError('[NON-UNIQUE KEY] graph already contains a node with that key.')
        
        self.adj[key] = []
    
    def insertEdge(self, key1, key2):
        """Inserts a new edge into the graph. If the edge already exists, does nothing.
        
        Args:
            key1: source node
            key2: destination node
        
        Raises:
            KeyError: An exception if key1, key2, or both, are not in the graph.
        
        Analysis:
            UpperBound: O(E)
        """
        if self.adj.get(key1) == None or self.adj.get(key2) == None:
            raise KeyError(f'[NON-EXISTENT KEY] cannot create an edge between non existent nodes. [key1:{key1}, key2:{key2}]')

        for k in self.adj[key1]:
            if k == key2:
                return

        self.adj[key1].append(key2)

    def deleteEdge(self, key1, key2):
        """Deletes edge from graph.
        
        Args:
            key1: Source vertex
            key2: Destination vertex.
        
        Raises:
            KeyError: An exception if key1 or key2 aren't present in the graph.
        
        Analysis:
            UpperBound: O(E)
        """
        if self.adj.get(key1) == None or self.adj.get(key2) == None:
            raise KeyError(f'[NON-EXISTENT KEY] cannot create an edge between non existent nodes.[key1:{key1}, key2:{key2}')

        try:
            self.adj[key1].remove(key2)
        except ValueError:
            return
    
    def adjacent(self, key):
        """Returns a list of keys adjacent to given key.
        
        Arg:
            key: a key representing a node in the graph.
        
        Raises:
            KeyError: an exception if the key is not valid.
        """

        if self.adj.get(key) == None:
            raise KeyError('key not found')
        
        return self.adj[key]
        
    def __iter__(self, source=None, style='depth-first'):
        """Returns an iterable of the graph which goes through every vertex.
        
        Args:
            source: source vertex
            style: command op for iteration style, must be one of:
                - 'depth-first'
                - 'breadth-first'  (Warning: May not visit every node.)
                - 'vertex-list'
        """

        if style == 'vertex-list':
            def h():
                for key in self.adj.keys():
                    yield key
            return h()

        if style == 'depth-first':
            
            def dfs():
                def dfs_start(n):
                    #color current vertex
                    colors[n] = 0
                    yield n
                    #go through neighbors of n
                    for neighbor in self.adj[n]:
                        if colors[neighbor] == 1:
                            yield from dfs_start(neighbor)
                    colors[n] = -1
                
                colors = {}
                #color each vertex white (set 1)
                for v in self.adj.keys():
                    colors[v] = 1

                for v in self.adj.keys():
                    if colors[v] == 1:
                        yield from dfs_start(v)

            return dfs()
        
        if style == 'breadth-first':
            if source == None:
                raise Exception('breadth-first search requires a source node.')
            from collections import deque

            def bfs():
                colors = {}
                colors[source] = 0
                for v in self.adj.keys():
                    colors[v] = 1

                queue = deque()
                queue.append(source)

                while queue:
                    n = queue.popleft()
                    yield n
                    for neighbor in self.adj[n]:
                        if colors[neighbor] == 1:
                            queue.append(neighbor)
                            colors[neighbor] = 0
                    colors[n] = -1
            return bfs()
    def minSpanTree(self, source):
        """Returns the minimum spanning tree of the graph, from source, as a dictionary.
        Args:
            source: a vertex in the graph
        Raises:
            KeyError: an exception if key is not a valid key
        Returns:
            D: a dictionary of (key,parent) values, where 'parent' is the next node up the chain in the min-span tree."""
        D = {}
        q = deque()
        q.append(source)
        colors = {}
        for v in self.adj.keys():
            colors[v] = 1
        colors[source] = 0
        while q:
            n = q.popleft()
            for neighbor in self.adj[n]:
                if colors[neighbor] == 1:
                    q.append(neighbor)
                    colors[neighbor] = 0
                    D[neighbor] = n
            colors[n] = -1
        return D
            
    def topologicalSort(self):
        """Returns a topological sort of the graph, if it exists. Returns [] otherwise."""
        order = []
        colors = {}
        def dfs(node):
            colors[node] = 0
            for neighbor in self.adj[node]:
                if colors[neighbor] == 1:
                    dfs(neighbor)
                elif colors[neighbor] == 0:
                    raise TypeError("[TypeError] topologicalSort only works on DAGs")
            colors[node] = -1
            order.append(node)

        #set all nodes to the color 'white'
        for n in self.adj.keys():
            colors[n] = 1

        for n in self.adj.keys():
            if colors[n] == 1:
                try:
                    dfs(n)
                except TypeError:
                    return []
        order.reverse()
        return order

class WeightedGraph(Graph):
    """A weighted graph is a normal graph who's edges also have an associated weight value."""
    def __init__(self, vertices=[], edges=[], weights=[]):
        """Initializes a new weighted graph."""
        Graph.__init__(self,vertices, edges)
        self.weights = {}

        for s,d,w in weights:
            self.insertEdge(s,d,w)

    def insert(self, key1=None, key2=None, weight=None):
        if key2==None:
            Graph.insertNode(self, key1)
        elif weight == None:
            self.insertEdge(key1, key2, 1)
        else:
            self.insertEdge(key1,key2,weight)

    def insertEdge(self, key1, key2, weight):
        """Inserts new edge into graph. Overwrites the existing edge if there is one. 
        Args:
            key1: key of source vertex.
            key2: key of destination vertex.
            weight: weight of the edge.
        Raises:
            ValueError: An exception if key1, key2, or both, are not in the graph.
        """
        #insert key1, key2 as normal.
        Graph.insertEdge(self, key1, key2)
        #insert weight as well
        self.weights[(key1,key2)] = weight
    
    def deleteEdge(self, key1, key2):
        """Deletes edge from graph.
        
        Args:
            key1: Source vertex
            key2: Destination vertex.
        
        Raises:
            KeyError: An exception if key1 or key2 aren't present in the graph.
        
        Analysis:
            UpperBound: O(E)
        """
        Graph.deleteEdge(self, key1, key2)
        self.weights[(key1, key2)] = None
    
    def bellmanFord(self, source):
        """Returns the min-path spanning tree as a dictionary.
        Args:
            source: a vertex in the graph
        Raises:
            KeyError: An exception if the key is not in the graph
            AttributeError: An exception if the graph has a negative weight cycle.
        Returns:
            A dictionary of (key, distance) values representing the distance from source to key.
        Analysis:
            O(|V| * |E|) runtime.
            bellman-Ford is a simple algorithm for finding the min-span tree of a general weighted graph. 
            Why does it work ?
                -Every shortest path is of the form <s, v1, v2, ... , d>
                
        """
    
        if self.adj.get(source) == None:
            raise KeyError(f"[NON-EXISTENT KEY] key {source} does not exist in the graph.")
            
        D = {}
        #init distances
        for v in self.adj.keys():
            D[v] = float('inf')
        D[source] = 0

        def relax(u,v):
            if D[v] > D[u] + self.weights.get((u,v)):
                D[v] = D[u] + self.weights.get((u,v))
        
        #do this process |V| - 1 times.
        for i in range(1, len(self.adj)):
            #iterate through edges
            for k in self.adj.keys():
                for d in self.adj[k]:
                    relax(k, d)

        #testing for negative cycle
        for k in self.adj.keys():
            for d in self.adj[k]:
                if D[d] > D[k] + self.weights.get((k,d)):
                    raise AttributeError("[NEGATIVE CYCLE] graphs with negative cycles don't have a well defined min-path tree.")
        #if all goes well, return the shorest-paths dict.
        return D

    def minPathTree(self, source):
        """Returns the min-path spanning tree from source as a dictionary. Assumes weights are non negative; not defined for negative weight graphs. May hang.
        Args:
            source: A vertex in the graph
        Raises:
            KeyError: An exception if the vertex is not in the graph.
        Returns:
            A dictionary of (key, distance) pairs representing the distance from source to key.
        Analysis:
            O(|V| * lg(|E|) + |E|)
        """
        if self.adj.get(source) == None:
            raise KeyError(f"[NON-EXISTENT KEY] key {source} does not exist in the graph.")

        D = {}
        for v in self.adj.keys():
            D[v] = float('inf')

        h = Heap(key=lambda x,y: x[0] < y[0]) #h contains all nodes that are currently under consideration for the next farthest away.
        h.insert((0, source))
        while len(h) != 0:
            #pop the minimum element off
            d, n = h.popMin()
            #If we're looking at a node that's already been seen before, we're done.
            if D.get(n) <= d:
                break
            #otherwise, we've found the shortest path to d.
            D[n] = d
            #append all neighbors to the heap
            for neighbor in self.adj[n]:
                h.insert((d + self.weights[(n, neighbor)], neighbor))
        return D

class Rational():
    """A class for computations on rational numbers."""
    def __init__(self, a, b):
        """Creates an instance of a rational number.
        Args: 
            a: integer numerator
            b: integer denominator
        Raises:
            ZeroDivisionError: An exception if b is 0.
            ValueError: An exception if a,b is not an integer."""
        if b == 0:
            raise ZeroDivisionError('Division by zero.')
        if not (type(a) == int and type(b) == int):
            raise ValueError('[INVALID ARGUMENTS] numerator / demoninator arguments must be integers.')

        _gcd = gcd(a,b)
        self.numerator = sign(a * b) * abs(a //_gcd)
        self.denominator = abs(b//_gcd)
 
    def __add__(self, n2):
        """Adds two rational numbers together. Returns a new rational number."""
        return Rational(self.numerator * n2.denominator + self.denominator * n2.numerator, self.denominator * n2.denominator)
    
    def __mul__(self, n2):
        if type(n2) == Rational:
            return Rational(self.numerator * n2.numerator, self.denominator * n2.denominator)
        elif type(n2) == int:
            return Rational(self.numerator * n2, self.denominator)
        else:
            return (self.numerator * n2) / self.denominator

    def _inverse(self):
        return Rational(self.denominator, self.numerator)

    def __truediv__(self, n2):
        #returns self/n2
        return self * (n2._inverse())
    
    def __sub__(self, n2):
        #returns self - n2
        return self + (n2 * -1)
    
    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

    def __eq__(self, n2):
        diff = self - n2
        return diff.numerator == 0

    def rationalize(n):
        """Class method. Takes a number and (attempts to) return a rational representation.
        Args:
            n: A rational number. 
        """
        sgn = sign(n)
        n = abs(n)

        d = 1
        while not fuzzy_equals(int(n), n):
            d = d * 10
            n = n * 10

        return Rational(sgn * int(n), d)

#Useful higher level function definitions +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def derivative(f, x):
    # finds the (approximate) slope value of the function f at x.
    # uses cauchy definition of convergence.
    e = 0.1
    x1 = x - e / 2
    x2 = x + e / 2

    m1 = (f(x2) - f(x1)) / e

    e = e/2
    x1 = x - e/2
    x2 = x + e/2

    m2 = (f(x2) - f(x1)) / e

    while abs(m1 - m2) > 0.00001:
        m1 = m2
        e = e/2
        x1 = x - e/2
        x2 = x + e/2
        m2 = (f(x2) - f(x1)) / e

    return m2

def inverse(f, guess = 1):
    #returns the inverse to f. Assumes that f is a strictly increasing function defined on 0 to infinity.
    #the inverse, g = f**(-1), is defined as such:
    #   g(x) = the value h such that f(h) = x 
    #

    #skeleton
    # get guess 
    #   if |f(guess) - n| < 0.00001:
    #       return guess
    #   if f(guess) < n:
    #       //guess higher
    #   if f(guess) > n:    
    #       // guess lower
    #
    def inv(x):
        guess = 1
        while guess > 0:
            t = f(guess)
            if abs(t - x) < 0.00001:
                return guess
            else:
                slope = derivative(f, guess)
                # find x = m*(guess2 - guess1) + f(guess)
                # guess2 = (x - f(guess)) / m + guess1
                guess = (x - t) / slope + guess
        raise Exception('Invalid function / Guess')
    return inv

def cache(fnc):
    """Caches a program. Easy solution to dynamic programming problems with simple recurrence relations.
    Args:
        fnc: A function
    NOTE: WIP
    """
    stored_answers = {}
    def helper(*params):
        if stored_answers.get(params):
            return stored_answers[params]
        else:
            ans = fnc(*params)
            stored_answers[params] = ans
            return ans
    return helper

def invert_error(f, error=Exception):
    #returns a function that throws an error when f DOESN'T throw an error, otherwise returns True.
    def h(*args) -> bool:
        try:
            f(*args)
        except:
            return True
        raise error
    return h

def create_graph(f, lst=[]):
    # generates the graph of f on the elements provided
    # elements of <lst> must be valid arguments to f.
    result = []
    for x in lst:
        result.append((x,f(x)))
    return result    

def create_inverse(graph):
    #Generates the inverse 'graph' of the graph given
    #<graph> is a list of input-output tuples.
    #Example: if <graph> is the graph of x**2, then <graph> might be [(-1, 1), (0,0), (1,1), (2,4), (3,9)]
    #         create_inverse(graph) = [(0,[0]), (1,[-1,1]), (4,[2]), (9,[3])]

    #skeleton
    #generate a dict struct whos keys are the outputs, and whos values are the inputs which lead to those outputs.
    d = {}
    for p in graph:
        if d.get(p[1]):
            d[p[1]].append(p[0])
        else:
            d[p[1]] = [p[0]]
    

    #return an ordered list of key/value pairs.
    """ What methods do dicts have again?"""
    key_vals = []
    keys = d.keys()
    for key in keys:
        key_vals.append((key, d[key]))

    key_vals.sort(key = lambda x: x[0])
    
    #return key_vals
    return key_vals

def plot(graph):
    import matplotlib.pyplot as plt
    plt.plot([x[0] for x in graph],[x[1] for x in graph])
    plt.show

def function_assert(f1, f2, args, equals = lambda x,y: x == y):
    #Same as 'assert', but with functions. 
    #asserts for each value given in args.
    #Contract:
    #   both f1, f2 accept the elements of args as arguments.
    #   equals accepts the ranges of f1 and f2.
    for a in args:
        try:
            assert equals(f1(a),f2(a))
        except AssertionError:
            print(f'AssertionError: argument {a}')
            return
    return

def timeit(f):
    """Creates a duplicate of f, but timed.
    Args:
        f: A function to be timed.
    """
    def h(*args):
        now = datetime.now()
        r = f(*args)
        then = datetime.now()

        print(f"Function {f} completed execution with arguments [{args}].")
        print(f"Function took: {(then - now).total_seconds()} s.")

        return r
        
    return h

# Useful mathematical function ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def char_func(n):
    #returns the characteristic function of n, ie: f(n)(x) = {1 if x=n, 0 if x!=n}
    def h(x):
        if x == n:
            return 1
        else:
            return 0

def bad_fib(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return bad_fib(n-1) + bad_fib(n-2)

@cache
def cache_fib(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return cache_fib(n-1) + cache_fib(n-2)

def fib(n):
    a = 1
    b = 0
    while n > 0:
        t = a
        a += b
        b = t
        n -= 1
    return a

def solve(f, error = 0.000001):
    #Solves equations of the form f(x) = 0, so that f(x) is within <error> of zero (default 1/1000000)
    #Contract:
    #   f a function which takes real numbers 
    #   f is monotone increasing or decreasing.
    #   f is defined AT zero, and for all values greater than 0.   

    #first guess
    ϵ = 1
    guess = 0

    while abs(f(guess)) > error:
        m = (f(guess + ϵ) - f(guess)) / ϵ
        b = f(guess) - m*guess
        guess = -b/m
        ϵ = ϵ/2
    return guess

def extend(graph):
    #Takes a graph and extends it to a real function buy connecting adjacent points to one another via lines. 
    #arguments outside the domain of the graph take on the value of the most adjacent point.
    lower_bound = graph[0][0]
    upper_bound = graph[-1][0]

    def extension(x):
        if x <= lower_bound:
            return graph[0][1]
        elif x >= upper_bound:
            return graph[-1][1]
        else:
            #NOTE have to find the domain values in g that are closest to x.
            #NOTE divide and conquer?
            l = 0
            r = len(graph) - 1

            while r > l + 1:
                m = (l + r) // 2

                if graph[m][0] >= x:
                    r = m
                if graph[m][0] <= x:
                    l = m

            return graph[l][1] + (x - math.floor(x)) * graph[r][1]

    return extension     

def gcd(a,b):
    """Uses euler's algorithm to calculate the gcd of 2 integers. 
    Args:
        a: an integer.
        b: an integer.
    Raises:
        ValueError: An exception if one or both of the arguments are not integers.
    """
    if not (type(a) == int and type(b) == int):
        raise ValueError('GCD expects integers.')

    a = abs(a)
    b = abs(b)
    a,b = max(a,b), min(a,b)

    while b != 0:
        r = a%b
        a,b = b,r
    return a

def sign(a):
    """Finds the sign of <a>. (1 if a is positive, -1 if a is negative.)
    Args:
        a: A number.
    """
    if a < 0:
        return -1
    return 1

# Useful list manipulation functions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def clean(lst):
    #creates a new list without redundant elements
    copy = lst.sorted()
    result = []
    last_unique = None
    for i,x in enumerate(copy):
        if x == last_unique:
            pass
        else:
            last_unique = x
            result.append(x)
    return result

def deepcopy(obj):
    # a deep copy function for an object
    # Note that this function was designed for deepcopy of lists, and the behaviour when applied to other objects is not gauranteed.
    if type(obj) == list:
        cpy = []
        for x in obj:
            cpy.append(deepcopy(x))
        return cpy
    else:
        return obj

def permutations(lst):
    #Takes a list and returns a list of every permutation of the list.
    #Contract:
    #   In: List is finitely terminating
    #   Out: List of permutations

    def distribute(el, l):
        #distributes an element, el, onto every element of l
        for x in l:
            x.append(el)
        return
    if lst == []:
        return [[]]
    else:
        r = permutations(lst[1:])
        a = deepcopy(r)
        distribute(lst[0], a)
        return r + a

def indexOf(lst, item):
    for x in range(0, len(lst)):
        if lst[x] == item:
            return x
    raise Exception('[INDEX NOT FOUND] list does not contain that element.')

#throws error if item exists in list already.
nindexOf = invert_error(indexOf, Exception('[DUPLICATE ELEMENT] list already contains that element.'))

def remove(lst, item):
    #removes item from list
    #if item is not in list, do nothing.

    try:
        i = indexOf(lst, item)
    except:
        return
    lst.pop(i)


#Useful stream definitions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mrange(start=0, stop=0,step=1):
    while start < stop:
        yield start
        start += step

def cross_product(stream1, stream2):
    #creates a stream that iterates over the cross product of the two streams.
    #There's more than one way to do this, so I might create more in the future, but for now I'll implement on that works on infinite streams.
    stream1_vals = []
    stream2_vals = []

    while True:
        #yield the next 2 values from the streams
        stream1_vals.append(next(stream1))
        stream2_vals.append(next(stream2))

        #iterate over the stored values so that their index-sum is constant the new s.
        for i in range(0, len(stream1_vals)):
            yield (stream1_vals[i], stream2_vals[-i-1])

def nth_product(*streams):
    #returns the cartesian product of all streams given, as a tuple
    if len(streams) == 1:
        while True:
            yield (next(streams[0]),)
    else:
        stream1 = streams[0]
        stream2 = nth_product(*streams[1:])

        stream1_vals = [] # x1,x2,x3...
        stream2_vals = [] # (y11,y12,y13..), (y21,y22,y23,...), ...
        while True: 
            stream1_vals.append(next(stream1))
            stream2_vals.append(list(next(stream2)))

            for x in range(0, len(stream1_vals)):
                next_val = stream2_vals[-x-1] + [stream1_vals[x]]
                yield tuple(next_val)
        
def tuplify(stream):
    while True:
        yield (next(stream),)
    
def repeat(stream):
    vals = list(stream)
    n = 0
    while True:
        yield vals[n]
        n = (n + 1) % len(vals)

def mask(stream1, stream2):
    repeat_seq = repeat(stream1)
    while True:
        if next(repeat_seq) == 1:
            yield next(stream2)
        else:
            next(stream2)
def natural_numbers():
    n = 0
    while True:
        yield n
        n += 1

def prepend_stream(val, stream):
    yield val
    yield from stream

def read_stream(nxt, delim):
	# utility function to read a stream up to (but not including) a delimiting sequence.
	# 'nxt' function which, when called, returns the next object in the stream. 
	# delim is the delimiting byte sequence.
	last = fla(len(delim))
	out = bytearray()

	while not last.equals(delim):
		a = last.push(nxt())
		if a != None:
			out.append(a)
	return out

#Useful functions for dealing with floating point numbers.
def fuzzy_equals(x,y, error = 0.000001):
    #determines if two real numbers are 'equal', or at least within <error> of being equal
    #Contract:
    #   x,y real
    #   error defaults to 1/1000000

    return abs(x-y) < error

if __name__ == "__main__":
    pass
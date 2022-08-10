#unit tests for mutility module.
from mutility import *
import random
import math
import heapq

def linkedlist_unittest():
    l1 = LinkedList(5,4,3,2,1)
    l2 = LinkedList(8,9,10)
    l3 = LinkedList()

    l1.insert(ListNode(0))
    l2.insert(ListNode(7))
    a = ListNode(8)
    l2.insert(a)
    l2.delete(a)

    assert str(l1) == '0,1,2,3,4,5'
    assert str(l2) == '7,10,9,8'
    assert str(l3) == ''
    assert l3.head() == l3.tail() == l3.nil

    #reverse
    l1.reverse()
    assert str(l1) == '5,4,3,2,1,0'
    l3.reverse()
    assert str(l3) == ''

    #union
    LinkedList.union(l2,l1)
    assert str(l2) == '7,10,9,8,5,4,3,2,1,0'
    LinkedList.union(l2,l3)
    assert str(l2) == '7,10,9,8,5,4,3,2,1,0'


def queue_unittest():
    q1 = Queue()
    
    q1.enqueue(1)
    q1.enqueue(2)
    assert q1.dequeue() == 1
    assert q1.dequeue() == 2

    try:
        q1.dequeue()
    except UnderflowError:
        assert True
    else:
        assert False
    print('Queue tests complete.')
    
def rational_unittest():
    #testing sign
    for i in range(-1, -11, -1):
        assert sign(i) == -1
    assert sign(0) == 1
    for i in range(1, 11, 1):
        assert sign(i) == 1

    
    #testing gcd
    for i in range(-100, 100):
        for j in range(-100, 100):
            assert gcd(i,j) == math.gcd(i,j)

    #testing rational numbers
    r1 = Rational(1,1) #1
    r2 = Rational(-1,1) #-1
    r3 = Rational(2,1) #2
    r4 = Rational(1,2) #1/2
    r5 = Rational(10, 100) #1/10
    r6 = Rational(56, 64) #7/8
    r7 = Rational(1, -1) #-1
    rationals = [r1, r2, r3 ,r4 ,r5 ,r6 ,r7]

    for n1 in rationals:
        for n2 in rationals:
            #additions
            n1 + n2
            #subtractions
            n1 - n2
            #multiplications
            n1 * n2
            #inversions
            n1._inverse()
            #divisions
            n1/n2
            #equality
            n1 == n2

    assert str(r1) == '1/1'
    assert str(r2) == '-1/1'
    assert str(r3) == '2/1'
    assert str(r4) == '1/2'
    assert str(r5) == '1/10'
    assert str(r6) == '7/8'
    assert str(r7) == '-1/1'

    assert Rational.rationalize(1) == Rational(1,1)
    assert Rational.rationalize(1.5) == Rational(3,2)
    assert Rational.rationalize(0.001) == Rational(1, 1000)
    assert Rational.rationalize(-1) == Rational(-1, 1)
    assert Rational.rationalize(0) == Rational(0, 1)

def cache_unittest():
    for i in range(100):
        print(timeit(cache_fib)(i))

def BST_unittest():
    
    #smash tests
    tree = BST()
    #insertions
    vals = []
    
    for i in range(10000):
        r = random.randint(0, 10000)
        vals.append(r)
        tree.insert(r)
        assert tree._verify()
    #lists
    for i in range(0,10000):
        l = random.randint(-100,10100)
        h = random.randint(-100, 10100)
        l,h = max(l,h), min(l,h)

        tree.list(l, h)

    for v in vals:
        tree.delete(v)
        assert tree._verify()
        
    #customtests
    tree = BST()
    tree.insert(1)
    tree.insert(2)
    tree.insert(0)
    tree.insert(-1)
    tree.insert(-0.5)

    assert sorted(tree.list(0, 1)) == [0, 1]
    assert sorted(tree.list(-1,1)) == [-1, -0.5, 0, 1]
    assert sorted(tree.list(1.5, 3)) == [2]
    assert sorted(tree.list(0,0)) == [0]
    assert sorted(tree.list(0.2, 0.5)) == []
    
    tree = BST()
    tree.insert(0)
    tree.insert(1)
    tree.insert(2)
    tree.insert(1.5)
    tree.insert(3)
    tree.insert(-1)
    tree.insert(-0.5)
    tree.insert(-0.25)
    tree.insert(-0.125)
    tree.insert(-0.0625)

    assert tree.successor(0) == 1
    assert tree.successor(1) == 1.5
    assert tree.successor(-0.0625) == 0
    assert tree.successor(3) == None
    try:
        tree.successor(-1000)
    except KeyError:
        pass
    assert tree.predecessor(0) == -0.0625
    assert tree.predecessor(-0.0625) == -0.125
    assert tree.predecessor(3) == 2
    assert tree.predecessor(-1) == None
    try:
        tree.predecessor(1000)
    except KeyError:
        pass

def AVLTree_unittest():
    #hail mary tests.
    
    tree = AVLTree()
    for i in range(10):
        s = str(tree)
        tree.insert(i)
        try:
            assert tree._verify()
        except AssertionError:
            print(f"AssertionError on insertion of [{i}]. Original:")
            print(s)
            print('After:')
            print(tree)
            return
    for i in range(10):
        s = str(tree)
        tree.delete(i)
        try:
            assert tree._verify()
        except AssertionError:
            print(f"AssertionError on deletion of [{i}]. Original:")
            print(s)
            print('After:')
            print(tree)
            return
 
    vals = []
    for i in range(1000):
        r = random.randint(0, 10000)
        vals.append(r)
        tree.insert(r)
        assert tree._verify()
    for v in vals:
        tree.delete(v)
        assert tree._verify()
    
    #custom tests

def graph_tests():
    print("Testing class Graph...")
    print("\tTesting class Node...")
    #vertex
    v1 = Node(1)
    v2 = Node(2)
    v3 = Node(1)

    assert v1 == v1
    assert v1 == v3
    assert v1 != v2
    assert hash(v1) == hash(v3)
    print("\tTesting core graph functionality (simple)...")
    g = Graph()
    g.insertNode(1)
    g.insertNode(2)
    g.insertNode(3)
    g.insertNode(4)
    g.insertEdge(1,2)
    g.insertEdge(2,3)
    g.insertEdge(3,4)
    g.insertEdge(4,2)

    #tests adjacent
    assert g.adjacent(4) == [2]
    
    iter1 = g.__iter__(style='vertex-list')
    iter2 = g.__iter__(style='depth-first')
    iter3 = g.__iter__(style='breadth-first', source=1)

    for v in [1,2,3,4]:
        assert v == next(iter1)
    for v in [1,2,3,4]:
        assert v == next(iter2)
    for v in [1,2,3,4]:
        assert v == next(iter3)  
    g.deleteEdge(4,2)
    assert 2 not in g.adjacent(4)

    #more complicated graph
    print("\tTesting core graph functionality (complex)...")
    g2 = Graph()
    g2.insert(1)
    g2.insert(2)
    g2.insert(3)
    g2.insert(4)
    g2.insert(5)
    g2.insert(6)

    g2.insert(1,2)
    g2.insert(2,5)
    g2.insert(5,4)
    g2.insert(4,2)
    g2.insert(1,4)
    g2.insert(3,5)
    g2.insert(3,6)
    g2.insert(6,6)

    iter4 = g2.__iter__()
    for v in [1,2,5,4,3,6]:
        assert v == next(iter4)

    print("\tTesting topologicalSort()...")
    g3 = Graph()
    g3.insert(0)
    g3.insert(1)
    g3.insert(2)
    g3.insert(3)
    g3.insert(4)
    g3.insert(5)
    g3.insert(6)
    g3.insertEdge(1,5)
    g3.insertEdge(2,5)
    g3.insertEdge(3,5)
    g3.insertEdge(1,2)
    g3.insertEdge(2,3)
    g3.insertEdge(3,4)
    g3.insertEdge(4,5)

    top = g3.topologicalSort()
    for v in g3.adj.keys():
        for neighbor in g3.adj[v]:
            assert indexOf(top, v) < indexOf(top, neighbor)

    g3.insertEdge(5,1)
    top = g3.topologicalSort()
    assert top == []

    print("\tTesting minSpanTree()...")
    g4 = Graph()
    g4.insert(1)
    g4.insert(2)
    g4.insert(3)
    g4.insert(4)
    g4.insert(5)
    g4.insert(6)
    g4.insert(1,2)
    g4.insert(1,3)
    g4.insert(3,4)
    g4.insert(2,4)
    g4.insert(4,1)
    g4.insert(1,1)
    g4.insert(5,4)
    g4.insert(6,5)
    g4.insert(2,6)
    g4.insert(0)
    minTree = g4.minSpanTree(1)
    assert minTree.get(1) == None
    assert minTree.get(2) == 1
    assert minTree.get(3) == 1
    assert minTree.get(4) == 2 or minTree.get(4) == 3
    assert minTree.get(5) == 6
    assert minTree.get(6) == 2
    assert minTree.get(0) == None
    print("Testing class WeightedGraph...")
    wg = WeightedGraph()
    print("\tTesting core graph functionality(simple)...")
    for i in range(1,7):
        wg.insert(i)
    wg.insert(1,2)
    wg.insert(1,3)
    wg.insert(3,4,4)
    wg.insert(2,4,3)
    wg.insert(2,5,-1)
    wg.insert(5,6,-1)
    wg.insert(5,4,2)
    wg.insert(0)
    wg.insert(5,1)
    assert 1 in wg.adjacent(5)
    wg.deleteEdge(5,1)
    assert 1 not in wg.adjacent(5)
    wg.insert(1,1,2)
    assert 1 in wg.adjacent(1)
    wg.deleteEdge(1,1)
    assert 1 not in wg.adjacent(1)
    print("\tTesting minPathTree()...")
    wg.insert(2,5,0)
    wg.insert(5,6,0)
    D = wg.minPathTree(1)
    assert D[0] == float('inf')
    assert D[1] == 0
    assert D[2] == 1
    assert D[3] == 1
    assert D[4] == 3
    assert D[5] == 1
    assert D[6] == 1
    print(D)
    print("Graph tests complete.")


def heap_tests():
    print("Testing Heap datastructure...")
    h = Heap()
    heap = []
    print("Attempting smash tests...")
    for i in range(1000):
        r = random.randint(0,10)
        h.insert(r)
        heapq.heappush(heap, r)
    
    for i in range(1000):
        try:
            a = h.popMin()
            b = heapq.heappop(heap)
            assert a == b
        except AssertionError:
            print("Failed on i = " + str(i) + ", for values (" + str(a)+ ', ' + str(b) + ').')
            return

    for i in range(10000):
        r = random.randint(-100, 1000000)
        h.insert(r)
        heapq.heappush(heap, r)
    
    for i in range(10000):
        try:
            a = h.popMin()
            b = heapq.heappop(heap)
            assert a == b
        except AssertionError:
            print("Failed on i = " + str(i) + ", for values (" + str(a)+ ', ' + str(b) + ').')
            return

    for i in range(100000):
        r = random.randint(-100, 10000000)
        h.insert(r)
        heapq.heappush(heap, r)
    
    for i in range(100000):
        try:
            a = h.popMin()
            b = heapq.heappop(heap)
            assert a == b
        except AssertionError:
            print("Failed on i = " + str(i) + ", for values (" + str(a)+ ', ' + str(b) + ').')
            return
    print("Tests succeeded. Exiting.")
def test_all():
    for obj in locals():
        if type(obj) == function:
            obj()

if __name__ == '__main__':
    graph_tests()
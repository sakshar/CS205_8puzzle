# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from heapq import heapify, heappop, heappush
import time


def evaluate(heuristic, node):
    # uniform cost search
    if heuristic == 1:
        return 0
    # A* with misplaced tiles
    elif heuristic == 2:
        match = np.count_nonzero(node == goal_st)
        if match == puzzle + 1:
            return 0
        return puzzle - match
    # A* with manhattan distance
    elif heuristic == 3:
        dist = 0
        for i in range(1, puzzle+1):
            pos = np.where(node == i)
            goal_pos = pos_map[i]
            dist += np.abs(pos[0][0] - goal_pos[0]) + np.abs(pos[1][0] - goal_pos[1])
        return dist
    return -1


def goal_test(node):
    if np.count_nonzero(node == goal_st) == puzzle + 1:
        return True
    return False


class State:
    def __init__(self, state):
        self.state = state
        self.parent = None
        self.gn = 0
        self.hn = np.inf
        self.fn = self.gn + self.hn

    def __lt__(self, other):
        return self.fn < other.fn

    def move_left(self, x, y):
        left_st = np.copy(self.state)
        left_st[x][y - 1], left_st[x][y] = left_st[x][y], left_st[x][y - 1]
        left = State(left_st)
        left.parent = self
        left.gn = self.gn+1
        left.hn = evaluate(heu, left_st)
        left.fn = left.gn + left.hn
        return left

    def move_right(self, x, y):
        right_st = np.copy(self.state)
        right_st[x][y + 1], right_st[x][y] = right_st[x][y], right_st[x][y + 1]
        right = State(right_st)
        right.parent = self
        right.gn = self.gn + 1
        right.hn = evaluate(heu, right_st)
        right.fn = right.gn + right.hn
        return right

    def move_up(self, x, y):
        up_st = np.copy(self.state)
        up_st[x - 1][y], up_st[x][y] = up_st[x][y], up_st[x - 1][y]
        up = State(up_st)
        up.parent = self
        up.gn = self.gn + 1
        up.hn = evaluate(heu, up_st)
        up.fn = up.gn + up.hn
        return up

    def move_down(self, x, y):
        down_st = np.copy(self.state)
        down_st[x + 1][y], down_st[x][y] = down_st[x][y], down_st[x + 1][y]
        down = State(down_st)
        down.parent = self
        down.gn = self.gn + 1
        down.hn = evaluate(heu, down_st)
        down.fn = down.gn + down.hn
        return down

    def expand(self):
        ind = np.where(self.state == 0)
        x, y = ind[0][0], ind[1][0]
        children = []
        if x == 0 and y == 0:
            children.append(self.move_down(x, y))
            children.append(self.move_right(x, y))
        elif x == 0 and y == self.state.shape[1] - 1:
            children.append(self.move_down(x, y))
            children.append(self.move_left(x, y))
        elif x == self.state.shape[0] - 1 and y == 0:
            children.append(self.move_up(x, y))
            children.append(self.move_right(x, y))
        elif x == self.state.shape[0] - 1 and y == self.state.shape[1] - 1:
            children.append(self.move_up(x, y))
            children.append(self.move_left(x, y))
        elif x == 0 and 0 < y < self.state.shape[1]:
            children.append(self.move_down(x, y))
            children.append(self.move_left(x, y))
            children.append(self.move_right(x, y))
        elif x == self.state.shape[0] - 1 and 0 < y < self.state.shape[1]:
            children.append(self.move_up(x, y))
            children.append(self.move_left(x, y))
            children.append(self.move_right(x, y))
        elif 0 < x < self.state.shape[0] and y == 0:
            children.append(self.move_down(x, y))
            children.append(self.move_up(x, y))
            children.append(self.move_right(x, y))
        elif 0 < x < self.state.shape[0] and y == self.state.shape[1] - 1:
            children.append(self.move_down(x, y))
            children.append(self.move_up(x, y))
            children.append(self.move_left(x, y))
        else:
            children.append(self.move_left(x, y))
            children.append(self.move_right(x, y))
            children.append(self.move_up(x, y))
            children.append(self.move_down(x, y))
        return children


# default parameters for 8-puzzle problem
sz = 3
puzzle = sz**2 - 1
heu = 3
init_st = np.array([[0, 7, 2],
                   [4, 6, 1],
                   [3, 5, 8]], dtype=int)
init = State(init_st)
goal_st = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 0]], dtype=int)
goal = State(goal_st)
pos_map = dict()
for i in range(1, puzzle+1):
    ind = np.where(goal_st == i)
    x, y = ind[0][0], ind[1][0]
    pos_map[i] = (x, y)


def a_star():
    nodes_expanded = 0
    heap = [init]
    heapify(heap)
    max_heap_size = 1
    while len(heap) > 0:
        current = heappop(heap)
        nodes_expanded = nodes_expanded + 1
        if goal_test(current.state):
            return current, nodes_expanded, max_heap_size
        children = current.expand()
        for ch in children:
            if current.parent is None or np.count_nonzero(ch.state == current.parent.state) < puzzle + 1:
                heappush(heap, ch)
        if max_heap_size < len(heap):
            max_heap_size = len(heap)
    return None, nodes_expanded, max_heap_size


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    result, nodes_expanded, max_heap_size = None, 0, 0
    print("Type input choice:\n1 -> default\n2 -> custom")
    choice = int(input())
    if choice == 1:
        #start = time.time()
        result, nodes_expanded, max_heap_size = a_star()
        #end = time.time()
        #print("Time elapsed: ", end - start)
    elif choice == 2:
        print("Now input elements separated by a space row-by-row")
        print("first row: ")
        r1 = input()
        print("second row: ")
        r2 = input()
        print("third row: ")
        r3 = input()
        init_st[0] = np.array([int(j) for j in r1.split()])
        init_st[1] = np.array([int(j) for j in r2.split()])
        init_st[2] = np.array([int(j) for j in r3.split()])
        print("Choose the heuristic function:\n1 -> Uniform Cost Search\n2 -> Misplaced Tiles\n3 -> Manhattan Distance")
        heu = int(input())
        result, nodes_expanded, max_heap_size = a_star()
    else:
        print("Invalid choice. Exiting!!!")
    if result is not None:
        solution_steps = []
        st = result
        while st is not None:
            solution_steps.append(st)
            st = st.parent
        print("Solution Depth: ", result.gn)
        print("Number of Nodes Expanded: ", nodes_expanded)
        print("Max Heap Size: ", max_heap_size)
        print("The solution steps are as follows:")
        for k in reversed(solution_steps):
            print("The best state to expand with g(n) = ", str(k.gn), "and h(n) = ", str(k.hn))
            print(k.state)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
import maze_helper as mh

# --- Các biến toàn cục ---
_directions_order = ['N', 'E', 'S', 'W']
_is_random = False
heuristic = None


def set_order(order_str=None, random=False):
    """Thiết lập thứ tự duyệt các hướng đi."""
    global _directions_order, _is_random
    _is_random = random
    if order_str:
        _directions_order = list(order_str)


def manhattan(state, goal):
    """Tính khoảng cách Manhattan giữa 2 điểm (x, y)."""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def min_index(lst):
    """Trả về index của phần tử nhỏ nhất trong list."""
    return min(range(len(lst)), key=lst.__getitem__)


# --- Cấu trúc Node ---
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1

    def __lt__(self, other):
        # Cần thiết cho Priority Queue (heapq)
        return self.path_cost < other.path_cost

    def path(self):
        """Trả về danh sách các state từ start đến node hiện tại."""
        node, path_back = self, []
        while node:
            path_back.append(node.state)
            node = node.parent
        return list(reversed(path_back))

    def actions(self):
        """Trả về danh sách các action từ start đến node hiện tại."""
        node, actions_back = self, []
        while node and node.parent:
            actions_back.append(node.action)
            node = node.parent
        return list(reversed(actions_back))


# --- Các hàm hỗ trợ Maze ---
def get_neighbors(maze, state):
    """Trả về danh sách các (action, next_state) hợp lệ."""
    x, y = state
    moves = {
        'N': (-1, 0),
        'E': (0, 1),
        'S': (1, 0),
        'W': (0, -1)
    }

    # Xác định thứ tự duyệt
    if _is_random:
        order = random.sample(_directions_order, len(_directions_order))
    else:
        order = _directions_order

    neighbors = []
    for action in order:
        dx, dy = moves[action]
        nx, ny = x + dx, y + dy
        # Kiểm tra biên và tường
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
            if maze[nx, ny] != 'X':
                neighbors.append((action, (nx, ny)))
    return neighbors


def reconstruct_result(node, reached, maze):
    """Đóng gói kết quả trả về cho notebook."""
    if node is None:
        return {'path': None, 'actions': None, 'reached': reached, 'maze_anim': []}

    path = node.path()
    actions = node.actions()

    # Tạo maze_anim để visualize
    maze_anim = []
    temp_maze = np.copy(maze)
    for r, c in reached:
        if temp_maze[r, c] == ' ':
            temp_maze[r, c] = '.'
    maze_anim.append(np.copy(temp_maze))

    # Vẽ đường đi
    for r, c in path:
        if temp_maze[r, c] == ' ' or temp_maze[r, c] == '.':
            temp_maze[r, c] = 'P'
    maze_anim.append(np.copy(temp_maze))

    return {
        'path': path,
        'actions': actions,
        'reached': reached,
        'maze_anim': maze_anim
    }


def show_path(maze, result):
    """Hiển thị maze với đường đi đã tìm được."""
    if result['path'] is None:
        print("No solution found!")
        mh.show_maze(maze)
        return

    temp_maze = np.copy(maze)

    # Đánh dấu các ô đã duyệt (.)
    for r, c in result['reached']:
        if temp_maze[r, c] == ' ':
            temp_maze[r, c] = '.'

    # Đánh dấu đường đi (P)
    for r, c in result['path']:
        if temp_maze[r, c] == ' ' or temp_maze[r, c] == '.':
            temp_maze[r, c] = 'P'

    mh.show_maze(temp_maze)


# --- Thuật toán Best-First Search (BFS, DFS, GBFS, A*) ---
def best_first_search(maze, strategy='BFS', W=1.0, debug=False, vis=False):
    start_state = mh.find_pos(maze, 'S')
    goal_state = mh.find_pos(maze, 'G')

    start_node = Node(start_state)

    if strategy == 'BFS' or strategy == 'DFS':
        frontier = [start_node]
        reached = {start_state}
    else:
        # GBFS hoặc A*
        h = heuristic(start_state, goal_state)
        if strategy == 'GBFS':
            priority = h
        else:  # A*
            priority = start_node.path_cost + W * h
        frontier = [(priority, 0, start_node)]
        reached = {start_state}

    reached_set = {start_state}
    explored_count = 0
    counter = 1  # Tie-breaker cho heap

    while frontier:
        if strategy == 'BFS':
            node = frontier.pop(0)  # FIFO
        elif strategy == 'DFS':
            node = frontier.pop()  # LIFO
        else:
            _, _, node = heapq.heappop(frontier)  # Min priority

        explored_count += 1

        if node.state == goal_state:
            if debug: print(f"Goal found! Explored: {explored_count}")
            return reconstruct_result(node, reached_set, maze)

        for action, next_state in get_neighbors(maze, node.state):
            child = Node(next_state, node, action, node.path_cost + 1)

            if next_state not in reached_set:
                reached_set.add(next_state)

                if strategy in ['BFS', 'DFS']:
                    frontier.append(child)
                else:
                    h = heuristic(next_state, goal_state)
                    if strategy == 'GBFS':
                        priority = h
                    else:  # A*
                        priority = child.path_cost + W * h
                    heapq.heappush(frontier, (priority, counter, child))
                    counter += 1

    if debug: print("No solution found.")
    return reconstruct_result(None, reached_set, maze)


# --- Thuật toán DFS (Depth-First Search) ---
def DFS(maze, limit=None, frontier_option=1, max_tries=100000, check_cycle=True, debug_reached=False, vis=False):
    start_state = mh.find_pos(maze, 'S')
    goal_state = mh.find_pos(maze, 'G')

    start_node = Node(start_state)

    # frontier_option 1: Dùng stack (LIFO), 2: Dùng đệ quy (hoặc mô phỏng)
    frontier = [start_node]
    reached = set()
    explored_count = 0
    tries = 0

    while frontier and tries < max_tries:
        tries += 1
        node = frontier.pop()

        if node.state == goal_state:
            if debug_reached: print(f"Goal found! Explored: {explored_count}")
            return reconstruct_result(node, reached, maze)

        if node.state not in reached or not check_cycle:
            reached.add(node.state)
            explored_count += 1

            if limit is not None and node.depth >= limit:
                continue

            for action, next_state in get_neighbors(maze, node.state):
                child = Node(next_state, node, action, node.path_cost + 1)

                # Kiểm tra cycle
                if check_cycle:
                    # Kiểm tra xem next_state có nằm trong path hiện tại không
                    in_path = False
                    temp = node
                    while temp:
                        if temp.state == next_state:
                            in_path = True
                            break
                        temp = temp.parent
                    if in_path:
                        continue

                frontier.append(child)

    if debug_reached: print(f"No solution found. Tries: {tries}")
    return reconstruct_result(None, reached, maze)


# --- Thuật toán IDS (Iterative Deepening Search) ---
def IDS(maze, frontier_option=2, max_tries=100000, vis=False):
    start_state = mh.find_pos(maze, 'S')
    goal_state = mh.find_pos(maze, 'G')

    # IDS chạy DFS với độ sâu tăng dần
    for depth in range(max_tries):
        result = DFS(maze, limit=depth, frontier_option=frontier_option,
                     max_tries=max_tries, check_cycle=True, debug_reached=False, vis=False)
        if result['path'] is not None:
            print(f"IDS found solution at depth {depth}")
            return result

    print("IDS: No solution found.")
    return reconstruct_result(None, set(), maze)
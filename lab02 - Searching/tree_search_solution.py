"""Tree-search implementations for the Maze Assignment (SGU - Tri Tue Nhan Tao).

API used by 02_Maze_Example.ipynb:
    ts.set_order(order / random=...)
    ts.best_first_search(maze, strategy="BFS"|"DFS"|"GBFS"|"A*", W=..., debug=..., vis=..., anim=...)
    ts.DFS(maze, check_cycle=..., limit=..., frontier_option=..., max_tries=..., debug_reached=..., vis=..., anim=...)
    ts.IDS(maze, frontier_option=..., max_tries=..., vis=..., anim=...)
    ts.show_path(maze, result) / ts.show_maze(maze)
    ts.heuristic = ts.manhattan
    ts.min_index(list)
Moi ham search tra ve dict: {'path', 'actions', 'reached', 'maze_anim', 'tries'}
"""
import heapq
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ----------------------------------------------------------------------
# Thu tu mo rong huong di
# ----------------------------------------------------------------------
DELTAS = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}
ORDER = list("NESW")
RANDOM = False


def set_order(order=None, random=False):
    """Dat thu tu duyet cac huong (vd "NESW") hooc che do random."""
    global ORDER, RANDOM
    RANDOM = random
    if order is not None:
        ORDER = list(order)
    if RANDOM:
        print("Directions are checked at every step in random order.")
    else:
        print(f"Directions are checked in the order {ORDER}")


# ----------------------------------------------------------------------
# Cac ham tien ich tren maze
# ----------------------------------------------------------------------
def find_pos(maze, what="S"):
    """Tra ve vi tri (x, y) dau tien cua ky tu `what` ('S' hooc 'G')."""
    pos = np.where(maze == what)
    return (pos[0][0], pos[1][0])


def look(maze, pos):
    """Doc ky tu tai o (x, y)."""
    x, y = pos
    return maze[x, y]


def _successors(maze, pos):
    """Cac nuoc di hop le tu `pos` theo thu tu hien tai (random neu RANDOM)."""
    order = ORDER[:]
    if RANDOM:
        random.shuffle(order)
    x, y = pos
    succ = []
    for a in order:
        dx, dy = DELTAS[a]
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] != 'X':
            succ.append((a, (nx, ny)))
    return succ


def manhattan(maze, pos):
    """Heuristic: khoang cach Manhattan tu pos den goal."""
    gx, gy = find_pos(maze, 'G')
    return abs(pos[0] - gx) + abs(pos[1] - gy)


def euclidean(maze, pos):
    """Heuristic: khoang cach Euclidean tu pos den goal."""
    gx, gy = find_pos(maze, 'G')
    return ((pos[0] - gx) ** 2 + (pos[1] - gy) ** 2) ** 0.5


heuristic = manhattan          # co gan lai: ts.heuristic = ts.manhattan


def min_index(values):
    """Chi muc cua phan tu nho nhat."""
    return min(range(len(values)), key=lambda i: values[i])


# ----------------------------------------------------------------------
# Ve hinh
# ----------------------------------------------------------------------
def show_maze(maze, fontsize=10):
    """Ve maze (giong maze_helper.show_maze)."""
    cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray', 'orange'])
    maze = np.copy(maze)
    start = find_pos(maze, 'S')
    goal = find_pos(maze, 'G')
    maze[maze == ' '] = 0
    maze[maze == 'X'] = 1
    maze[maze == 'S'] = 2
    maze[maze == 'G'] = 3
    maze[maze == 'P'] = 4
    maze[maze == '.'] = 5
    maze[maze == 'F'] = 6
    maze = maze.astype(int)
    fig, ax = plt.subplots()
    ax.imshow(maze, cmap=cmap, norm=colors.BoundaryNorm(list(range(cmap.N + 1)), cmap.N))
    plt.text(start[1], start[0], "S", fontsize=fontsize, color="white",
             horizontalalignment='center', verticalalignment='center')
    plt.text(goal[1], goal[0], "G", fontsize=fontsize, color="white",
             horizontalalignment='center', verticalalignment='center')
    plt.show()


def show_path(maze, result, fontsize=10):
    """In thong tin va ve duong di (P) + cac o da duyet (.) tu ket qua search."""
    if result['path'] is not None:
        print(f"Path length: {len(result['path']) - 1}")
    print(f"Reached squares: {len(result['reached'])}")
    m = np.copy(maze)
    for pos in result['reached']:
        if m[pos] == ' ':
            m[pos] = '.'
    if result['path'] is not None:
        for pos in result['path']:
            if m[pos] in (' ', '.'):
                m[pos] = 'P'
    show_maze(m, fontsize)


def _snapshot(maze, explored, frontier_set, current):
    """Mot frame de animation: '.' = da duyet, 'F' = frontier, 'P' = o dang mo rong."""
    m = np.copy(maze)
    for pos in explored:
        if m[pos] == ' ':
            m[pos] = '.'
    for pos in frontier_set:
        if m[pos] in (' ', '.'):
            m[pos] = 'F'
    if current is not None and m[current] not in ('S', 'G'):
        m[current] = 'P'
    return m


def _display_anim(result):
    """Hien animation neu co maze_anim (dung maze_helper.animate_maze)."""
    try:
        from maze_helper import animate_maze
        import IPython.display as display
        display.display(animate_maze(result))
    except Exception:
        pass


def _reconstruct(parent, state):
    """Lui theo parent de tra ve (path, actions)."""
    path = [state]
    actions = []
    while parent[state] is not None:
        p, a = parent[state]
        actions.append(a)
        path.append(p)
        state = p
    path.reverse()
    actions.reverse()
    return path, actions


def _new_result():
    return {'path': None, 'actions': None, 'reached': set(), 'maze_anim': None, 'tries': 0}


# ----------------------------------------------------------------------
# Best-first search (BFS / DFS / GBFS / A* / Weighted A*)
# ----------------------------------------------------------------------
def best_first_search(maze, strategy="BFS", W=1, debug=False, vis=False, anim=False):
    """Best-first search tong quat.

    strategy: "BFS" (f=g), "DFS" (f=-g), "GBFS" (f=h), "A*" (f=g+W*h).
    Tie-break: node moi duoc them gan nhat (de giu huong di).
    """
    start = find_pos(maze, 'S')
    goal = find_pos(maze, 'G')
    use_h = strategy in ("GBFS", "A*")

    def f_value(g, h):
        if strategy == "BFS":
            return g
        if strategy == "DFS":
            return -g
        if strategy == "GBFS":
            return h
        if strategy == "A*":
            return g + W * h
        raise ValueError(f"Unknown strategy: {strategy}")

    counter = 0
    gmap = {start: 0}
    parent = {start: None}
    reached = {start}
    explored = set()
    frontier_set = {start}
    heap = [(f_value(0, heuristic(maze, start) if use_h else 0), -counter, start)]
    maze_anim = [] if (anim or vis) else None
    result = _new_result()

    while heap:
        f, _, state = heapq.heappop(heap)
        frontier_set.discard(state)
        explored.add(state)
        result['tries'] += 1
        if maze_anim is not None:
            maze_anim.append(_snapshot(maze, explored, frontier_set, state))
        if debug:
            print(f"Expand {state} f={f} frontier_size={len(heap)}")
        if state == goal:
            result['path'], result['actions'] = _reconstruct(parent, state)
            break
        g = gmap[state]
        for a, nxt in _successors(maze, state):
            if nxt in reached:                      # cycle checking (graph search)
                continue
            reached.add(nxt)
            frontier_set.add(nxt)
            parent[nxt] = (state, a)
            gmap[nxt] = g + 1
            counter += 1
            hn = heuristic(maze, nxt) if use_h else 0
            heapq.heappush(heap, (f_value(g + 1, hn), -counter, nxt))

    result['reached'] = reached
    if maze_anim is not None:
        result['maze_anim'] = maze_anim
    if vis:
        _display_anim(result)
    return result


# ----------------------------------------------------------------------
# Depth-first search (stack, LIFO)
# ----------------------------------------------------------------------
def DFS(maze, check_cycle=True, limit=None, frontier_option=1,
        max_tries=100000, debug_reached=False, vis=False, anim=False):
    """DFS dung stack (LIFO).

    check_cycle:      chan cycle bang reached.
    limit:            gioi han sau (depth-limited DFS), None = khong gioi han.
    frontier_option:  1 = chi check cac o da mo rong;
                      2 = check ca cac o dang nam trong frontier (frontier + explored).
    max_tries:        so lan mo rong toi da truoc khi bo cuoc (tra ve path=None).
    debug_reached:    True thi result['reached'] chua cac o da duyet (de ve vung xam).
    """
    start = find_pos(maze, 'S')
    goal = find_pos(maze, 'G')
    parent = {start: None}
    seen = {start}
    explored = set()
    frontier_set = {start}
    stack = [(start, 0)]
    maze_anim = [] if (anim or vis) else None
    result = _new_result()

    while stack and result['tries'] < max_tries:
        state, depth = stack.pop()
        frontier_set.discard(state)
        explored.add(state)
        result['tries'] += 1
        if maze_anim is not None:
            maze_anim.append(_snapshot(maze, explored, frontier_set, state))
        if state == goal:
            result['path'], result['actions'] = _reconstruct(parent, state)
            break
        if limit is not None and depth >= limit:
            continue
        # LIFO: day vao theo thu tu -> lay ra theo thu tu dao nguoc
        for a, nxt in _successors(maze, state):
            if check_cycle:
                if frontier_option == 2:
                    if nxt in seen:
                        continue
                else:
                    if nxt in explored:
                        continue
                seen.add(nxt)
            if nxt not in parent:
                parent[nxt] = (state, a)
            stack.append((nxt, depth + 1))
            frontier_set.add(nxt)

    result['reached'] = explored if debug_reached else set()
    if maze_anim is not None:
        result['maze_anim'] = maze_anim
    if vis:
        _display_anim(result)
    return result


# ----------------------------------------------------------------------
# Iterative deepening search
# ----------------------------------------------------------------------
def IDS(maze, frontier_option=2, max_tries=100000, vis=False, anim=False):
    """IDS: lap lai depth-limited DFS voi limit tang dan 0, 1, 2, ..."""
    max_limit = maze.shape[0] * maze.shape[1]
    result = _new_result()
    for limit in range(0, max_limit + 1):
        result = DFS(maze, check_cycle=True, limit=limit,
                     frontier_option=frontier_option, max_tries=max_tries,
                     anim=anim)
        if result['path'] is not None:
            break
    result['reached'] = set()          # IDS khong giu reached (giong ban goc)
    if vis:
        _display_anim(result)
    return result


# khoi dong: thu tu mac dinh (in ra 1 lan khi import, giong notebook goc)
set_order("NESW")
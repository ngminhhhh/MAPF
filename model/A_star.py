import heapq
import numpy as np

def in_bounds(H, W, r, c):
    return 0 <= r < H and 0 <= c < W

def passable(grid, r, c):
    return grid[r, c] == 0

def manhattan(r, c, gr, gc):
    return abs(r - gr) + abs(c - gc)

def A_star(grid: np.ndarray, start:tuple[int, int], goal:tuple[int, int]):
    '''
    @return:
        path - List[Tuple[int, int]]: path from start to goal compute by A* 
    '''
    H, W = grid.shape
    sr, sc = start
    gr, gc = goal

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    open_heap = []
    heapq.heappush(open_heap, (manhattan(sr, sc, gr, gc), 0, (sr, sc))) # * (f, g, node)

    par = {(sr, sc) : None}
    g_score = {(sr, sc) : 0}

    while open_heap:
        f, g, (r, c) = heapq.heappop(open_heap)

        if (r, c) == (gr, gc):
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = par[cur]

            path.reverse()
            return path
        
        for (dr, dc) in deltas:
            nr, nc = r + dr, c + dc
            if not in_bounds(H, W, nr, nc) or not passable(grid, nr, nc): continue

            tentative_g = g + 1

            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                g_score[(nr, nc)] = tentative_g
                par[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (tentative_g + manhattan(nr, nc, gr, gc), tentative_g, (nr, nc)))

    return None
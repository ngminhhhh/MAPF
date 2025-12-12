from heapq import heappush, heappop
from itertools import count
import json
import re
from tqdm import tqdm
import time  # NEW


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(pos, grid):
    h, w = len(grid), len(grid[0])
    r, c = pos
    moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
            yield (nr, nc)


def build_constraint_table(constraints, agent_id):
    vertex = {}
    edge = {}
    for ct in constraints:
        if ct['agent'] not in (agent_id, -1):
            continue
        t = ct['time']
        if ct['type'] == 'vertex':
            vertex.setdefault(t, set()).add(tuple(ct['pos']))
        else:  # edge
            edge.setdefault(t, set()).add((tuple(ct['from']), tuple(ct['to'])))
    return vertex, edge


def is_constrained(curr, nxt, t, vertex, edge):
    if t in vertex and nxt in vertex[t]:
        return True

    if (t - 1) in edge and (curr, nxt) in edge[t - 1]:
        return True
    return False


def a_star(agent_id, start, goal, grid, constraints, deadline=None):
    """
    A* với constraint; nếu vượt deadline (wall-clock) thì trả về None.
    """
    vertex, edge = build_constraint_table(constraints, agent_id)

    open_list = []
    g_score = {}
    parents = {}

    start_state = (start, 0)  # (position, time)
    g_score[start_state] = 0
    parents[start_state] = None

    heappush(open_list, (manhattan(start, goal), 0, start_state))

    max_constraint_t = max(vertex.keys(), default=0)

    visited = set()
    while open_list:
        # check timeout
        if deadline is not None and time.time() > deadline:
            return None

        f, g, (pos, t) = heappop(open_list)
        if (pos, t) in visited:
            continue
        visited.add((pos, t))

        if pos == goal and t >= max_constraint_t:
            # reconstruct path
            path = []
            cur = (pos, t)
            while cur is not None:
                path.append(cur[0])
                cur = parents[cur]
            return list(reversed(path))

        nt = t + 1
        for nxt in neighbors(pos, grid):
            if is_constrained(pos, nxt, nt, vertex, edge):
                continue
            state = (nxt, nt)
            ng = g + 1
            if state in g_score and ng >= g_score[state]:
                continue
            g_score[state] = ng
            parents[state] = (pos, t)
            heappush(open_list, (ng + manhattan(nxt, goal), ng, state))

    return None


def detect_conflict(paths):
    n = len(paths)
    max_len = max(len(p) for p in paths)
    padded = []

    for p in paths:
        if len(p) < max_len:
            p = p + [p[-1]] * (max_len - len(p))
        padded.append(p)

    # vertex conflict
    for t in range(max_len):
        occ = {}
        for i in range(n):
            pos = padded[i][t]
            if pos in occ:
                return {
                    'a1': occ[pos],
                    'a2': i,
                    'time': t,
                    'type': 'vertex',
                    'pos': pos
                }
            occ[pos] = i

    # edge conflict (swap)
    for t in range(max_len - 1):
        for i in range(n):
            for j in range(i + 1, n):
                if padded[i][t] == padded[j][t + 1] and padded[i][t + 1] == padded[j][t]:
                    return {
                        'a1': i,
                        'a2': j,
                        'time': t + 1,
                        'type': 'edge',
                        'pos1': padded[i][t],
                        'pos2': padded[i][t + 1],
                    }

    return None


def cbs(instance, max_time=1.0):
    start_time = time.time()
    deadline = start_time + max_time

    grid = instance['grid']
    starts = [tuple(p) for p in instance['starts']]
    goals = [tuple(p) for p in instance['goals']]
    n = len(starts)

    # Root node
    root_constraints = []
    root_paths = []
    for i in range(n):
        # check timeout trước khi gọi A*
        if time.time() > deadline:
            return None

        p = a_star(i, starts[i], goals[i], grid, root_constraints, deadline=deadline)
        if p is None:
            return None
        root_paths.append(p)
    root_cost = sum(len(p) - 1 for p in root_paths)  # sum-of-costs

    open_list = []
    counter = count()
    heappush(open_list, (root_cost, next(counter), {
        'constraints': root_constraints,
        'paths': root_paths,
        'cost': root_cost
    }))

    while open_list:
        # check timeout mỗi vòng lặp
        if time.time() > deadline:
            return None

        _, _, node = heappop(open_list)
        paths = node['paths']

        conflict = detect_conflict(paths)
        if conflict is None:
            return paths

        for agent in (conflict['a1'], conflict['a2']):
            # check timeout trước khi đi sâu thêm
            if time.time() > deadline:
                return None

            new_constraints = list(node['constraints'])

            if conflict['type'] == 'vertex':
                new_constraints.append({
                    'agent': agent,
                    'type': 'vertex',
                    'time': conflict['time'],
                    'pos': conflict['pos'],
                })
            else:  # edge conflict
                if agent == conflict['a1']:
                    from_pos, to_pos = conflict['pos1'], conflict['pos2']
                else:
                    from_pos, to_pos = conflict['pos2'], conflict['pos1']
                new_constraints.append({
                    'agent': agent,
                    'type': 'edge',
                    'time': conflict['time'] - 1,
                    'from': from_pos,
                    'to': to_pos,
                })

            # Re-plan cho agent đó
            new_paths = list(paths)
            new_p = a_star(agent, starts[agent], goals[agent], grid, new_constraints, deadline=deadline)
            if new_p is None:
                continue

            new_paths[agent] = new_p
            new_cost = sum(len(p) - 1 for p in new_paths)

            heappush(open_list, (new_cost, next(counter), {
                'constraints': new_constraints,
                'paths': new_paths,
                'cost': new_cost
            }))

    return None


if __name__ == "__main__":
    n_instances = 1000
    MAX_TIME_PER_INSTANCE = 10

    success_count = 0
    fail_count = 0

    for i in tqdm(range(n_instances), desc="Solving CBS instances"):
        with open(f"./data/10x10-10/train/instance_{i}.json", "r") as f:
            instance = json.load(f)

        paths = cbs(instance, max_time=MAX_TIME_PER_INSTANCE)
        if paths is None:
            print(f"\nInstance {i}: no solution (timeout or unsolvable)")
            fail_count += 1
            continue

        success_count += 1

        solution = {
            "paths": [
                [[r, c] for (r, c) in p]
                for p in paths
            ]
        }

        json_str = json.dumps(solution, indent=4, ensure_ascii=False)
        json_str = re.sub(
            r'\[\s*((?:-?\d+\s*,\s*)*-?\d+\s*)\]',
            lambda m: '[' + ', '.join(x.strip() for x in m.group(1).replace('\n', ' ').split(',')) + ']',
            json_str
        )

        with open(f"./data/10x10-10/train-solution/solution_{i}.json", "w") as f:
            f.write(json_str)

    print(f"Total instances: {n_instances}")
    print(f"Success: {success_count}")
    print(f"Fail (timeout/unsolvable): {fail_count}")


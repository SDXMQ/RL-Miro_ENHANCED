import numpy as np
import random
import heapq

def generate_maze_dfs(height, width):
    """DFS 백트래킹으로 미로를 생성합니다.
    height, width는 실제 격자 크기 (짝수일 경우 +1로 보정).
    생성 후 경로 보장 및 루프 추가로 복잡도를 높입니다.
    반환: numpy 배열 (0=길, 1=벽, 2=시작, 3=목표)
    """
    # 홀수 크기로 보정 (벽+길 패턴)
    h = height if height % 2 == 1 else height + 1
    w = width if width % 2 == 1 else width + 1

    maze = np.ones((h, w), dtype=int)

    # DFS 시작점 (1,1)
    start = (1, 1)
    maze[start] = 0
    stack = [start]
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        current = stack[-1]
        random.shuffle(directions)
        found = False
        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if 1 <= nr < h - 1 and 1 <= nc < w - 1 and maze[nr, nc] == 1:
                # 벽 뚫기
                maze[current[0] + dr // 2, current[1] + dc // 2] = 0
                maze[nr, nc] = 0
                stack.append((nr, nc))
                found = True
                break
        if not found:
            stack.pop()

    # 원래 요청 크기에 맞게 잘라내기
    sliced_maze = maze[:height, :width].copy()

    # 짝수 크기일 때 테두리 벽(1) 채워주고 두꺼운 벽 보정
    if height % 2 == 0:
        sliced_maze[height - 1, :] = 1
        for c in range(1, width - 1):
            if sliced_maze[height - 2, c] == 1 and sliced_maze[height - 3, c] == 0:
                # 2x2 통로(로비) 형성 여부 확인 후 깎기
                lobby_created = False
                for dc in [-1, 0]:
                    sub_c = c + dc
                    if 0 <= sub_c < width - 1:
                        if (sliced_maze[height - 3, sub_c] != 1 and 
                            sliced_maze[height - 3, sub_c + 1] != 1 and
                            (sub_c == c or sliced_maze[height - 2, sub_c] != 1) and
                            (sub_c + 1 == c or sliced_maze[height - 2, sub_c + 1] != 1)):
                            lobby_created = True
                            break
                if not lobby_created:
                    sliced_maze[height - 2, c] = 0
    if width % 2 == 0:
        sliced_maze[:, width - 1] = 1
        for r in range(1, height - 1):
            if sliced_maze[r, width - 2] == 1 and sliced_maze[r, width - 3] == 0:
                # 2x2 통로(로비) 형성 여부 확인 후 깎기
                lobby_created = False
                for dr in [-1, 0]:
                    sub_r = r + dr
                    if 0 <= sub_r < height - 1:
                        if (sliced_maze[sub_r, width - 3] != 1 and
                            sliced_maze[sub_r + 1, width - 3] != 1 and
                            (sub_r == r or sliced_maze[sub_r, width - 2] != 1) and
                            (sub_r + 1 == r or sliced_maze[sub_r + 1, width - 2] != 1)):
                            lobby_created = True
                            break
                if not lobby_created:
                    sliced_maze[r, width - 2] = 0

    # 시작점, 목표점 설정
    start_r, start_c = 1, 1
    goal_r = height - 2 if height % 2 == 1 else height - 3
    goal_c = width - 2 if width % 2 == 1 else width - 3
    sliced_maze[start_r, start_c] = 2
    sliced_maze[goal_r, goal_c] = 3

    # --- 경로 보장: 가중치 BFS(다익스트라)로 시작→목적지 경로 확보 ---
    _ensure_path(sliced_maze, start_r, start_c, goal_r, goal_c)

    # --- 복잡도 향상: 내부 벽 무작위 허물기로 루프(우회로) 추가 ---
    _add_loops(sliced_maze, loop_ratio=0.04)

    # --- 로비 현상 해소: 2x2 통로(로비)를 연결성을 유지하며 벽으로 메우기 ---
    _remove_lobbies(sliced_maze, start_r, start_c, goal_r, goal_c)

    # --- 두꺼운 벽 해소: 2x2 이상의 벽 뭉치를 찾아 다듬기 ---
    _remove_thick_walls(sliced_maze)

    return sliced_maze


def _ensure_path(maze, sr, sc, gr, gc):
    """다익스트라 알고리즘으로 시작점→목적지 최단 경로를 찾고,
    경로 상의 벽(1)을 모두 통로(0)로 변환하여 탈출구를 보장합니다.
    통로 이동 비용=1, 벽 허물기 비용=100000 (최대한 벽을 안 뚫도록 유도)."""
    h, w = maze.shape
    INF = float('inf')
    dist = np.full((h, w), INF)
    prev = np.full((h, w, 2), -1, dtype=int)
    dist[sr, sc] = 0

    # (비용, r, c) 최소 힙
    pq = [(0, sr, sc)]
    adj = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while pq:
        cost, r, c = heapq.heappop(pq)
        if cost > dist[r, c]:
            continue
        if r == gr and c == gc:
            break
        for dr, dc in adj:
            nr, nc = r + dr, c + dc
            # 외곽 테두리(행 0, 열 0, 마지막 행/열)는 벽 유지 — 뚫으면 안 됨
            if nr <= 0 or nr >= h - 1 or nc <= 0 or nc >= w - 1:
                continue
            tile = maze[nr, nc]
            # 시작(2), 목적지(3), 길(0)은 비용 1 / 벽(1)은 비용 100000
            move_cost = 1 if tile != 1 else 100000
            new_dist = cost + move_cost
            if new_dist < dist[nr, nc]:
                dist[nr, nc] = new_dist
                prev[nr, nc] = [r, c]
                heapq.heappush(pq, (new_dist, nr, nc))

    # 경로 역추적하며 벽(1) → 통로(0) 변환
    r, c = gr, gc
    while not (r == sr and c == sc):
        pr, pc = prev[r, c]
        if pr == -1:
            break  # 도달 불가 (이론상 발생하지 않음)
        if maze[r, c] == 1:
            maze[r, c] = 0
        r, c = pr, pc


def _add_loops(maze, loop_ratio=0.07):
    """테두리 외곽을 제외한 내부 벽 중 품질 제약 조건을 충족하는 벽만
    선택적으로 허물어 우회로(루프)를 만듭니다."""
    h, w = maze.shape
    # 내부 벽 후보지
    inner_walls = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if maze[r, c] == 1:
                inner_walls.append((r, c))

    random.shuffle(inner_walls)
    
    max_to_remove = max(1, int(len(inner_walls) * loop_ratio))
    removed_count = 0

    for r, c in inner_walls:
        if removed_count >= max_to_remove:
            break

        # 1. 상하좌우 이웃 확인 (0이 아닌 시작/목표/보너스 타일 포함)
        neighbors = [
            (r - 1, c), # 상
            (r + 1, c), # 하
            (r, c - 1), # 좌
            (r, c + 1)  # 우
        ]
        
        path_neighbors = []
        for nr, nc in neighbors:
            if 0 <= nr < h and 0 <= nc < w:
                if maze[nr, nc] != 1:
                    path_neighbors.append((nr, nc))

        # 정확히 2개의 통로 이웃과 연결되어야 하며, 그 이웃이 마주보는 직선 방향이어야 함 (교차로 확장 방지)
        if len(path_neighbors) != 2:
            continue
            
        (r1, c1), (r2, c2) = path_neighbors[0], path_neighbors[1]
        is_vertical = (c1 == c2 == c) and (abs(r1 - r2) == 2)
        is_horizontal = (r1 == r2 == r) and (abs(c1 - c2) == 2)
        
        if not (is_vertical or is_horizontal):
            continue

        # 2. 2x2 통로(로비) 형성 여부 확인
        lobby_detected = False
        for dr in [-1, 0]:
            for dc in [-1, 0]:
                sub_r, sub_c = r + dr, c + dc
                if 0 <= sub_r < h - 1 and 0 <= sub_c < w - 1:
                    # 임시로 (r,c)를 0으로 바꿀 때 2x2 영역 내 벽 개수
                    wall_count = 0
                    for tr in range(2):
                        for tc in range(2):
                            curr_r, curr_c = sub_r + tr, sub_c + tc
                            if curr_r == r and curr_c == c:
                                continue
                            if maze[curr_r, curr_c] == 1:
                                wall_count += 1
                    if wall_count == 0:
                        lobby_detected = True
                        break
            if lobby_detected:
                break
        
        if lobby_detected:
            continue

        # 3. 1x1 고립벽(섬)이 생기는지 확인 (도넛 현상 방지)
        isolated_wall_created = False
        for nr, nc in neighbors:
            if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] == 1:
                adj_paths = 0
                for nnr, nnc in [(nr-1, nc), (nr+1, nc), (nr, nc-1), (nr, nc+1)]:
                    if 0 <= nnr < h and 0 <= nnc < w:
                        if (nnr == r and nnc == c) or maze[nnr, nnc] != 1:
                            adj_paths += 1
                if adj_paths == 4:
                    isolated_wall_created = True
                    break
        
        if isolated_wall_created:
            continue

        # 모든 조건 만족 시 벽 제거
        maze[r, c] = 0
        removed_count += 1


def _remove_lobbies(maze, sr, sc, gr, gc):
    """미로 내부에 2x2 크기 이상의 통로(모두 1이 아닌 셀)가 존재할 경우,
    시작점(2)과 목적지(3) 간의 연결성을 유지하면서
    해당 2x2 통로 중 적절한 셀 하나를 벽(1)으로 메워 로비 현상을 해소합니다."""
    h, w = maze.shape
    
    def is_connected():
        visited = np.zeros((h, w), dtype=bool)
        queue = [(sr, sc)]
        visited[sr, sc] = True
        
        head = 0
        while head < len(queue):
            r, c = queue[head]
            head += 1
            if r == gr and c == gc:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not visited[nr, nc] and maze[nr, nc] != 1:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        return False

    changed = True
    while changed:
        changed = False
        for r in range(h - 1):
            for c in range(w - 1):
                # 2x2 영역이 모두 통로(1이 아님)인지 확인
                if (maze[r, c] != 1 and maze[r+1, c] != 1 and
                    maze[r, c+1] != 1 and maze[r+1, c+1] != 1):
                    
                    candidates = [(r, c), (r+1, c), (r, c+1), (r+1, c+1)]
                    random.shuffle(candidates)
                    
                    for cr, cc in candidates:
                        if (cr == sr and cc == sc) or (cr == gr and cc == gc):
                            continue
                        if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                            continue
                        
                        # 새로 생긴 벽이 사방이 통로인 1x1 고립벽이 되는지 검사
                        adj_walls = 0
                        for nr, nc in [(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)]:
                            if 0 <= nr < h and 0 <= nc < w:
                                if maze[nr, nc] == 1:
                                    adj_walls += 1
                        if adj_walls == 0:
                            continue
                        
                        # 새로 생긴 벽이 주변과 합쳐져 2x2 벽 블록(두꺼운 벽)을 형성하는지 검사
                        thick_wall_detected = False
                        for dr in [-1, 0]:
                            for dc in [-1, 0]:
                                sub_r, sub_c = cr + dr, cc + dc
                                if 0 <= sub_r < h - 1 and 0 <= sub_c < w - 1:
                                    wall_count = 0
                                    for tr in range(2):
                                        for tc in range(2):
                                            curr_r, curr_c = sub_r + tr, sub_c + tc
                                            if curr_r == cr and curr_c == cc:
                                                wall_count += 1
                                            elif maze[curr_r, curr_c] == 1:
                                                wall_count += 1
                                    if wall_count == 4:
                                        thick_wall_detected = True
                                        break
                            if thick_wall_detected:
                                break
                        if thick_wall_detected:
                            continue
                        
                        original_val = maze[cr, cc]
                        maze[cr, cc] = 1
                        
                        if is_connected():
                            changed = True
                            break
                        else:
                            maze[cr, cc] = original_val
                    
                    if changed:
                        break
            if changed:
                continue


def _remove_thick_walls(maze):
    """미로 내부에 2x2 크기 이상의 두꺼운 벽(모두 1인 영역)이 존재할 경우,
    외곽 테두리를 해치지 않고 다른 품질 제약을 어기지 않으면서
    벽의 두께를 1칸으로 줄이기 위해 일부 벽을 통로(0)로 전환합니다."""
    h, w = maze.shape
    changed = True
    
    while changed:
        changed = False
        for r in range(1, h - 2):
            for c in range(1, w - 2):
                # 2x2 영역이 모두 벽(1)인지 확인
                if (maze[r, c] == 1 and maze[r+1, c] == 1 and
                    maze[r, c+1] == 1 and maze[r+1, c+1] == 1):
                    
                    candidates = [(r, c), (r+1, c), (r, c+1), (r+1, c+1)]
                    best_cell = None
                    max_adjacent_paths = -1
                    
                    for cr, cc in candidates:
                        if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                            continue
                            
                        adj_paths = 0
                        for nr, nc in [(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)]:
                            if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] != 1:
                                adj_paths += 1
                        
                        # 2x2 통로 형성 방지
                        lobby_detected = False
                        for dr in [-1, 0]:
                            for dc in [-1, 0]:
                                sub_r, sub_c = cr + dr, cc + dc
                                if 0 <= sub_r < h - 1 and 0 <= sub_c < w - 1:
                                    wall_count = 0
                                    for tr in range(2):
                                        for tc in range(2):
                                            curr_r, curr_c = sub_r + tr, sub_c + tc
                                            if curr_r == cr and curr_c == cc:
                                                continue
                                            if maze[curr_r, curr_c] == 1:
                                                wall_count += 1
                                    if wall_count == 0:
                                        lobby_detected = True
                                        break
                            if lobby_detected:
                                break
                                
                        if lobby_detected:
                            continue
                            
                        # 고립벽 형성 방지
                        isolated_wall_created = False
                        for nr, nc in [(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)]:
                            if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] == 1:
                                adj_of_neighbor = 0
                                for nnr, nnc in [(nr-1, nc), (nr+1, nc), (nr, nc-1), (nr, nc+1)]:
                                    if 0 <= nnr < h and 0 <= nnc < w:
                                        if (nnr == cr and nnc == cc) or maze[nnr, nnc] != 1:
                                            adj_of_neighbor += 1
                                if adj_of_neighbor == 4:
                                    isolated_wall_created = True
                                    break
                        if isolated_wall_created:
                            continue
                            
                        if adj_paths > max_adjacent_paths:
                            max_adjacent_paths = adj_paths
                            best_cell = (cr, cc)
                            
                    if best_cell is not None:
                        br, bc = best_cell
                        maze[br, bc] = 0
                        changed = True
                        break



import numpy as np
import random

def generate_maze_dfs(height, width):
    """DFS 백트래킹으로 미로를 생성합니다.
    height, width는 실제 격자 크기 (짝수일 경우 +1로 보정).
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

    # 시작점, 목표점 설정
    maze[1, 1] = 2
    # 목표점: 우하단 근처의 열린 칸 찾기
    goal_r, goal_c = h - 2, w - 2
    maze[goal_r, goal_c] = 3

    # 원래 요청 크기에 맞게 잘라내기
    return maze[:height, :width]

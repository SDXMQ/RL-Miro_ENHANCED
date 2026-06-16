import numpy as np

class MazeEnv:
    """
    미로 찾기 강화학습을 위한 환경 클래스 (Gym 스타일 API 차용)
    """
    # 이동 방향 인덱스: 0(상, w), 1(하, s), 2(좌, a), 3(우, d)
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, maze_map=None):
        # 기본 미로 정의
        if maze_map is None:
            self.original_map = np.array([
                [1,2,1,1,1,1,1,1,1,1], # ■▽■■■■■■■■
                [1,0,0,0,0,0,0,0,0,1], # ■□□□□□□□□■
                [1,0,1,1,0,1,1,1,1,1], # ■□■■□■■■■■
                [1,0,1,1,0,0,0,0,0,1], # ■□■■□□□□□■
                [1,0,1,1,1,1,1,1,0,1], # ■□■■■■■■□■
                [1,0,0,0,0,0,0,1,0,1], # ■□□□□□□■□■
                [1,0,1,1,1,0,1,1,0,1], # ■□■■■□■■□■
                [1,0,0,0,1,0,0,1,1,1], # ■□□□■□□■■■
                [1,0,1,0,1,1,0,0,0,1], # ■□■□■■□□□■
                [1,1,1,1,1,1,1,1,3,1], # ■■■■■■■■▼■
            ])
        else:
            self.original_map = np.array(maze_map, dtype=int)

        self.height, self.width = self.original_map.shape
        
        # 시작점(2)과 목적지(3) 좌표 찾기
        start_indices = np.where(self.original_map == 2)
        if len(start_indices[0]) > 0:
            self.start_pos = (start_indices[0][0], start_indices[1][0])
        else:
            raise ValueError("미로에 시작점(2)이 존재하지 않습니다.")

        goal_indices = np.where(self.original_map == 3)
        if len(goal_indices[0]) > 0:
            self.goal_pos = (goal_indices[0][0], goal_indices[1][0])
        else:
            raise ValueError("미로에 목적지(3)가 존재하지 않습니다.")

        # 현재 상태 (좌표)
        self.current_pos = self.start_pos

    def reset(self):
        """
        환경을 초기 상태로 리셋하고 시작 위치를 반환합니다.
        """
        self.current_pos = self.start_pos
        return self.current_pos

    def is_valid_position(self, x, y):
        """
        주어진 좌표가 미로 격자 내부이며 벽(1)이 아닌지 검사합니다.
        """
        return 0 <= x < self.height and 0 <= y < self.width and self.original_map[x, y] != 1

    def get_valid_actions(self, state):
        """
        특정 상태(x, y)에서 취할 수 있는 유효한(벽이 아닌) 행동 인덱스 목록을 반환합니다.
        """
        x, y = state
        valid_actions = []
        for idx, (dx, dy) in enumerate(self.ACTIONS):
            if self.is_valid_position(x + dx, y + dy):
                valid_actions.append(idx)
        return valid_actions

    def step(self, action_idx):
        """
        에이전트에게 행동을 수행하게 하고, 다음 상태, 보상, 종료 여부를 반환합니다.
        """
        if action_idx < 0 or action_idx >= len(self.ACTIONS):
            raise ValueError(f"유효하지 않은 행동 인덱스: {action_idx}")

        dx, dy = self.ACTIONS[action_idx]
        x, y = self.current_pos
        nx, ny = x + dx, y + dy

        # 벽에 부딪히거나 범위를 벗어나는 경우 제자리에 멈추고 감점 페널티
        if not self.is_valid_position(nx, ny):
            reward = -10
            done = False
            return self.current_pos, reward, done

        # 정상적인 길로 이동
        self.current_pos = (nx, ny)
        
        # 보상 설계
        if self.current_pos == self.goal_pos:
            reward = 1000
            done = True
        else:
            reward = -10
            done = False

        return self.current_pos, reward, done

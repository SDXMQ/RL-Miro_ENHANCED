import numpy as np
import pygame
import sys
import random
from environment import MazeEnv
from agent import RLAgent

# ──────────────────────────────────────────────
# 색상 테마 정의 (HEX -> RGB)
# ──────────────────────────────────────────────
COLOR_BG = (30, 30, 36)          # #1e1e24 - 딥 차콜
COLOR_SIDEBAR = (42, 42, 53)     # #2a2a35 - 다크 블루그레이
COLOR_WALL = (63, 75, 92)        # #3f4b5c - 스틸 블루
COLOR_PATH = (18, 18, 20)        # #121214 - 다크 블랙(길)
COLOR_START = (6, 214, 160)      # #06d6a0 - 네온 민트 (시작)
COLOR_GOAL = (255, 93, 115)      # #ff5d73 - 네온 핑크 (도착)
COLOR_AGENT = (255, 209, 102)    # #ffd166 - 네온 골드 (에이전트)
COLOR_TRAP = (220, 50, 50)       # 함정 - 어두운 레드
COLOR_TRAP_OFF = (100, 50, 50)   # 함정 비활성 - 더 어두운 레드
COLOR_BONUS = (80, 160, 255)     # 보너스 - 밝은 블루
COLOR_BONUS_OFF = (50, 70, 90)   # 보너스 수집됨 - 음소거 블루
COLOR_TEXT_MAIN = (255, 255, 255)
COLOR_TEXT_MUTED = (160, 160, 180)
COLOR_GRID_LINE = (35, 35, 45)

# 버튼 색상
BTN_COLOR = (49, 130, 206)       # 파랑
BTN_HOVER = (66, 153, 225)
BTN_DANGER = (229, 62, 62)       # 빨강
BTN_DANGER_HOVER = (245, 101, 101)
BTN_SUCCESS = (56, 161, 105)     # 초록
BTN_SUCCESS_HOVER = (72, 187, 120)
BTN_WARN = (214, 158, 46)        # 노랑/오렌지
BTN_WARN_HOVER = (236, 181, 63)
BTN_PURPLE = (128, 90, 213)
BTN_PURPLE_HOVER = (159, 122, 234)

# ──────────────────────────────────────────────
# UI 위젯 클래스
# ──────────────────────────────────────────────
class Button:
    def __init__(self, x, y, w, h, text, color, hover_color, text_color=(255, 255, 255), action=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.action = action
        self.is_hovered = False

    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (255, 255, 255, 40), self.rect, width=1, border_radius=8)

        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered and self.action:
                self.action()
                return True
        return False


class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, start_val, label, is_float=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = start_val
        self.label = label
        self.is_float = is_float
        self.grabbed = False

        self.handle_w = 14
        self.handle_rect = pygame.Rect(0, 0, self.handle_w, h + 8)
        self.update_handle_pos()

    def update_handle_pos(self):
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        self.handle_rect.centerx = self.rect.x + int(ratio * self.rect.width)
        self.handle_rect.centery = self.rect.centery

    def draw(self, screen, font):
        val_text = f"{self.val:.3f}" if self.is_float else f"{int(self.val)}"
        label_surf = font.render(f"{self.label}: {val_text}", True, COLOR_TEXT_MUTED)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 22))

        # 트랙
        pygame.draw.rect(screen, (60, 60, 75), self.rect, border_radius=4)
        # 채워진 부분
        filled_w = self.handle_rect.centerx - self.rect.x
        if filled_w > 0:
            filled_rect = pygame.Rect(self.rect.x, self.rect.y, filled_w, self.rect.height)
            pygame.draw.rect(screen, COLOR_START, filled_rect, border_radius=4)

        # 핸들
        color = COLOR_AGENT if self.grabbed else (240, 240, 245)
        pygame.draw.rect(screen, color, self.handle_rect, border_radius=4)

    def handle_event(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect.collidepoint(mouse_pos) or self.rect.collidepoint(mouse_pos):
                self.grabbed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.grabbed = False

        if self.grabbed and event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            rel_x = max(0, min(mouse_pos[0] - self.rect.x, self.rect.width))
            ratio = rel_x / self.rect.width
            self.val = self.min_val + ratio * (self.max_val - self.min_val)
            if not self.is_float:
                self.val = int(round(self.val))
            self.update_handle_pos()
            return True
        return False


# ──────────────────────────────────────────────
# DFS 미로 생성기
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# 메인 애플리케이션 클래스
# ──────────────────────────────────────────────
class MazeApp:
    # 프로그램 실행 상태 상수
    STATE_IDLE = 0
    STATE_TRAINING = 1
    STATE_FAST_TRAINING = 2
    STATE_TESTING = 3
    STATE_EDITING = 4

    # 에디터 브러시 종류
    BRUSH_WALL = 1
    BRUSH_PATH = 0
    BRUSH_TRAP = 4
    BRUSH_BONUS = 5
    BRUSH_START = 2
    BRUSH_GOAL = 3

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("RL Maze Solver - Enhanced")

        self.screen_w = 1220
        self.screen_h = 780
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.clock = pygame.time.Clock()

        # 폰트 초기화 (한글 지원을 위해 기본 시스템 폰트 로드 시도)
        font_candidates = ["malgungothic", "applesdgothicneo", "nanumgothic", "arial", "helvetica"]
        self.font = None
        for fc in font_candidates:
            try:
                self.font = pygame.font.SysFont(fc, 14)
                self.font_title = pygame.font.SysFont(fc, 20, bold=True)
                self.font_big = pygame.font.SysFont(fc, 28, bold=True)
                self.font_small = pygame.font.SysFont(fc, 12)
                break
            except Exception:
                continue
        if self.font is None:
            self.font = pygame.font.Font(None, 20)
            self.font_title = pygame.font.Font(None, 28)
            self.font_big = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 16)

        # 환경 및 에이전트 초기화
        self.env = MazeEnv()
        self.algorithm = 'q_learning'
        self.agent = RLAgent(
            state_space_shape=(self.env.height, self.env.width),
            alpha=0.15, gamma=0.85, epsilon=0.999,
            algorithm=self.algorithm
        )

        self.state = self.env.reset()
        self.current_state = MazeApp.STATE_IDLE

        # 메트릭 관리
        self.episode_count = 0
        self.step_count = 0
        self.success_count = 0
        self.best_steps = float('inf')
        self.history_steps = []  # 에피소드별 이동 횟수

        # 미로 렌더링 영역
        self.maze_rect = pygame.Rect(20, 20, 600, 600)
        self._update_cell_size()

        # 파티클 효과용 목록
        self.particles = []

        # Q-value 시각화 스위치
        self.show_q_overlay = True

        # 에디터 브러시 타입
        self.editor_brush = MazeApp.BRUSH_WALL

        # Sarsa용 다음 행동 캐시
        self._sarsa_next_action = None

        # 학습 결과 기록용 로그 파일 설정
        self.log_filepath = "rl_training.log"
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write("\n=== 학습 시뮬레이션 세션 시작 ===\n")
            f.write("Episode,Steps,SuccessRate,Alpha,Gamma,Epsilon,Algorithm,MazeSize\n")

        # UI 요소 생성
        self.setup_ui()

    def _update_cell_size(self):
        self.cell_w = self.maze_rect.width // self.env.width
        self.cell_h = self.maze_rect.height // self.env.height

    # ──────────────────────────────────────────
    # UI 구성
    # ──────────────────────────────────────────
    def setup_ui(self):
        sx = 650  # sidebar x 시작
        bw = 170   # 버튼 기본 너비
        bw2 = 350  # 풀 너비

        # --- 알고리즘 선택 버튼 (토글) ---
        algo_y = 100
        self.algo_buttons = [
            Button(sx, algo_y, 110, 30, "Q-Learning", BTN_SUCCESS, BTN_SUCCESS_HOVER, action=lambda: self.set_algorithm('q_learning')),
            Button(sx + 120, algo_y, 110, 30, "SARSA", BTN_COLOR, BTN_HOVER, action=lambda: self.set_algorithm('sarsa')),
            Button(sx + 240, algo_y, 110, 30, "Double-Q", BTN_PURPLE, BTN_PURPLE_HOVER, action=lambda: self.set_algorithm('double_q')),
        ]

        # --- 함정 토글 버튼 ---
        self.trap_toggle_btn = Button(sx, 140, bw2, 28, "함정: ON", BTN_DANGER, BTN_DANGER_HOVER, action=self.toggle_trap)

        # --- 슬라이더 ---
        slider_y = 190
        self.sliders = [
            Slider(sx, slider_y, bw2, 10, 0.01, 1.0, self.agent.alpha, "학습률 (Alpha)", is_float=True),
            Slider(sx, slider_y + 48, bw2, 10, 0.0, 1.0, self.agent.gamma, "감가율 (Gamma)", is_float=True),
            Slider(sx, slider_y + 96, bw2, 10, 1, 100, 30, "학습 속도 (FPS)"),
            Slider(sx, slider_y + 144, bw2, 10, 5, 30, self.env.height, "미로 크기 (Size)"),
        ]

        # --- 시뮬레이션 제어 버튼 ---
        ctrl_y = 380
        self.buttons = [
            Button(sx, ctrl_y, bw, 34, "▶ 학습 시작", BTN_SUCCESS, BTN_SUCCESS_HOVER, action=self.start_training),
            Button(sx + 180, ctrl_y, bw, 34, "⏸ 일시 정지", BTN_COLOR, BTN_HOVER, action=self.pause_training),
            Button(sx, ctrl_y + 42, bw, 34, "⚡ 고속학습 100", BTN_COLOR, BTN_HOVER, action=self.start_fast_training),
            Button(sx + 180, ctrl_y + 42, bw, 34, "🧪 테스트", BTN_COLOR, BTN_HOVER, action=self.start_testing),
            Button(sx, ctrl_y + 84, bw2, 34, "🔄 에이전트 및 Q-Table 초기화", BTN_DANGER, BTN_DANGER_HOVER, action=self.reset_all),
        ]

        # --- 에디터/유틸 버튼 ---
        edit_y = ctrl_y + 130
        self.edit_buttons = [
            Button(sx, edit_y, bw, 34, "🖊 편집 모드", BTN_WARN, BTN_WARN_HOVER, action=self.toggle_edit_mode),
            Button(sx + 180, edit_y, bw, 34, "🎲 랜덤 미로", BTN_PURPLE, BTN_PURPLE_HOVER, action=self.generate_random_maze),
            Button(sx, edit_y + 42, bw, 34, "💾 저장", BTN_COLOR, BTN_HOVER, action=self.save_model),
            Button(sx + 180, edit_y + 42, bw, 34, "📂 불러오기", BTN_COLOR, BTN_HOVER, action=self.load_model),
        ]

        # --- 에디터 브러시 버튼 (편집 모드일 때만 표시) ---
        brush_y = edit_y + 84
        self.brush_buttons = [
            Button(sx, brush_y, 68, 28, "벽", COLOR_WALL, (90, 100, 120), action=lambda: self._set_brush(1)),
            Button(sx + 72, brush_y, 68, 28, "길", (60, 60, 70), (90, 90, 100), action=lambda: self._set_brush(0)),
            Button(sx + 144, brush_y, 68, 28, "함정", COLOR_TRAP, BTN_DANGER_HOVER, action=lambda: self._set_brush(4)),
            Button(sx + 216, brush_y, 68, 28, "보너스", COLOR_BONUS, (100, 180, 255), action=lambda: self._set_brush(5)),
            Button(sx + 288, brush_y, 68, 28, "시작", COLOR_START, (30, 240, 180), action=lambda: self._set_brush(2)),
        ]

    # ──────────────────────────────────────────
    # 버튼 액션 핸들러
    # ──────────────────────────────────────────
    def set_algorithm(self, algo):
        """알고리즘 전환 시 Q-Table 초기화 후 새 에이전트 생성"""
        self.algorithm = algo
        self.agent = RLAgent(
            state_space_shape=(self.env.height, self.env.width),
            alpha=self.sliders[0].val,
            gamma=self.sliders[1].val,
            epsilon=0.999,
            algorithm=algo
        )
        self.state = self.env.reset()
        self.current_state = MazeApp.STATE_IDLE
        self.episode_count = 0
        self.step_count = 0
        self.success_count = 0
        self.best_steps = float('inf')
        self.history_steps = []
        self.particles.clear()
        self._sarsa_next_action = None

    def toggle_trap(self):
        self.env.trap_enabled = not self.env.trap_enabled
        if self.env.trap_enabled:
            self.trap_toggle_btn.text = "함정: ON"
            self.trap_toggle_btn.color = BTN_DANGER
            self.trap_toggle_btn.hover_color = BTN_DANGER_HOVER
        else:
            self.trap_toggle_btn.text = "함정: OFF"
            self.trap_toggle_btn.color = (100, 60, 60)
            self.trap_toggle_btn.hover_color = (130, 80, 80)

    def start_training(self):
        if self.current_state in (MazeApp.STATE_IDLE, MazeApp.STATE_TESTING, MazeApp.STATE_EDITING):
            self.current_state = MazeApp.STATE_TRAINING
            if self.env.current_pos == self.env.goal_pos:
                self.state = self.env.reset()
                self.step_count = 0
            # Sarsa: 첫 행동 미리 선택
            if self.algorithm == 'sarsa':
                valid = self.env.get_valid_actions(self.state)
                self._sarsa_next_action = self.agent.get_action(self.state, valid)

    def pause_training(self):
        if self.current_state in (MazeApp.STATE_TRAINING, MazeApp.STATE_FAST_TRAINING):
            self.current_state = MazeApp.STATE_IDLE

    def start_fast_training(self):
        self.current_state = MazeApp.STATE_FAST_TRAINING
        self.run_fast_training_loop(100)
        self.current_state = MazeApp.STATE_IDLE

    def start_testing(self):
        self.current_state = MazeApp.STATE_TESTING
        self.state = self.env.reset()
        self.step_count = 0

    def reset_all(self):
        new_size = int(self.sliders[3].val)
        # 기본 빈 미로 맵 생성 (테두리는 벽, (1,1) 시작, (new_size-2, new_size-2) 도착)
        new_map = np.zeros((new_size, new_size), dtype=int)
        new_map[0, :] = 1
        new_map[-1, :] = 1
        new_map[:, 0] = 1
        new_map[:, -1] = 1
        new_map[1, 1] = 2
        new_map[new_size - 2, new_size - 2] = 3

        self.env.set_map(new_map)
        self._update_cell_size()

        self.agent = RLAgent(
            state_space_shape=(self.env.height, self.env.width),
            alpha=self.sliders[0].val,
            gamma=self.sliders[1].val,
            epsilon=0.999,
            algorithm=self.algorithm
        )
        self.state = self.env.reset()
        self.current_state = MazeApp.STATE_IDLE
        self.episode_count = 0
        self.step_count = 0
        self.success_count = 0
        self.best_steps = float('inf')
        self.history_steps = []
        self.particles.clear()
        self._sarsa_next_action = None

    def toggle_edit_mode(self):
        if self.current_state == MazeApp.STATE_EDITING:
            self.current_state = MazeApp.STATE_IDLE
        else:
            self.current_state = MazeApp.STATE_EDITING
            self.state = self.env.reset()
            self.step_count = 0

    def generate_random_maze(self):
        """DFS 알고리즘으로 랜덤 미로 생성 후 환경에 적용"""
        new_size = int(self.sliders[3].val)
        new_map = generate_maze_dfs(new_size, new_size)
        self.env.set_map(new_map)
        self._update_cell_size()
        # 에이전트 Q-table 리셋 (맵 크기가 바뀔 수 있으므로)
        self.agent = RLAgent(
            state_space_shape=(self.env.height, self.env.width),
            alpha=self.sliders[0].val,
            gamma=self.sliders[1].val,
            epsilon=0.999,
            algorithm=self.algorithm
        )
        self.state = self.env.reset()
        self.current_state = MazeApp.STATE_IDLE
        self.episode_count = 0
        self.step_count = 0
        self.success_count = 0
        self.best_steps = float('inf')
        self.history_steps = []
        self.particles.clear()

    def _set_brush(self, brush_type):
        self.editor_brush = brush_type

    def save_model(self):
        """현재 맵과 Q-Table을 .npz 파일로 저장"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.asksaveasfilename(
                title="모델 저장",
                defaultextension=".npz",
                filetypes=[("NumPy Compressed", "*.npz"), ("All Files", "*.*")]
            )
            root.destroy()
            if filepath:
                save_data = {
                    'maze_map': self.env.original_map,
                    'algorithm': np.array([self.algorithm], dtype=object),
                    'trap_enabled': np.array([self.env.trap_enabled]),
                    'alpha': np.array([self.agent.alpha]),
                    'gamma': np.array([self.agent.gamma]),
                    'epsilon': np.array([self.agent.epsilon]),
                    'episode_count': np.array([self.episode_count]),
                    'success_count': np.array([self.success_count]),
                    'best_steps': np.array([self.best_steps if self.best_steps != float('inf') else -1]),
                }
                if self.algorithm == 'double_q':
                    save_data['q_table_a'] = self.agent.q_table_a
                    save_data['q_table_b'] = self.agent.q_table_b
                else:
                    save_data['q_table'] = self.agent._q_table
                np.savez(filepath, **save_data)
        except Exception as e:
            print(f"저장 실패: {e}")

    def load_model(self):
        """저장된 .npz 파일에서 맵과 Q-Table 복원"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(
                title="모델 불러오기",
                filetypes=[("NumPy Compressed", "*.npz"), ("All Files", "*.*")]
            )
            root.destroy()
            if filepath:
                data = np.load(filepath, allow_pickle=True)
                # 맵 복원
                self.env.set_map(data['maze_map'])
                self._update_cell_size()
                # 알고리즘 복원
                self.algorithm = str(data['algorithm'][0])
                # 함정 상태 복원
                self.env.trap_enabled = bool(data['trap_enabled'][0])
                if self.env.trap_enabled:
                    self.trap_toggle_btn.text = "함정: ON"
                    self.trap_toggle_btn.color = BTN_DANGER
                    self.trap_toggle_btn.hover_color = BTN_DANGER_HOVER
                else:
                    self.trap_toggle_btn.text = "함정: OFF"
                    self.trap_toggle_btn.color = (100, 60, 60)
                    self.trap_toggle_btn.hover_color = (130, 80, 80)
                # 에이전트 복원
                self.agent = RLAgent(
                    state_space_shape=(self.env.height, self.env.width),
                    alpha=float(data['alpha'][0]),
                    gamma=float(data['gamma'][0]),
                    epsilon=float(data['epsilon'][0]),
                    algorithm=self.algorithm
                )
                if self.algorithm == 'double_q':
                    self.agent.q_table_a = data['q_table_a']
                    self.agent.q_table_b = data['q_table_b']
                else:
                    self.agent._q_table = data['q_table']
                # 슬라이더 동기화
                self.sliders[0].val = self.agent.alpha
                self.sliders[0].update_handle_pos()
                self.sliders[1].val = self.agent.gamma
                self.sliders[1].update_handle_pos()
                self.sliders[3].val = self.env.height
                self.sliders[3].update_handle_pos()
                # 메트릭 복원
                self.episode_count = int(data['episode_count'][0])
                self.success_count = int(data['success_count'][0])
                bs = float(data['best_steps'][0])
                self.best_steps = float('inf') if bs < 0 else bs
                self.history_steps = []
                # 상태 리셋
                self.state = self.env.reset()
                self.current_state = MazeApp.STATE_IDLE
                self.step_count = 0
                self.particles.clear()
        except Exception as e:
            print(f"불러오기 실패: {e}")

    # ──────────────────────────────────────────
    # 에이전트 파라미터 동기화
    # ──────────────────────────────────────────
    def update_agent_params(self):
        """슬라이더 설정 값을 에이전트에 반영합니다."""
        self.agent.alpha = self.sliders[0].val
        self.agent.gamma = self.sliders[1].val

    # ──────────────────────────────────────────
    # 파티클 시스템
    # ──────────────────────────────────────────
    def spawn_celebration_particles(self):
        goal_x = self.maze_rect.x + self.env.goal_pos[1] * self.cell_w + self.cell_w // 2
        goal_y = self.maze_rect.y + self.env.goal_pos[0] * self.cell_h + self.cell_h // 2
        for _ in range(30):
            color = random.choice([COLOR_START, COLOR_AGENT, (255, 100, 200)])
            speed_x = random.uniform(-4, 4)
            speed_y = random.uniform(-6, -1)
            life = random.randint(30, 60)
            size = random.randint(3, 7)
            self.particles.append([goal_x, goal_y, speed_x, speed_y, life, color, size])

    def update_particles(self):
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[3] += 0.15
            p[4] -= 1
            if p[4] <= 0:
                self.particles.remove(p)

    def draw_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), p[6])

    # ──────────────────────────────────────────
    # 학습 / 테스트 로직
    # ──────────────────────────────────────────
    def step_training(self):
        """1단계 학습 업데이트를 수행합니다."""
        valid_actions = self.env.get_valid_actions(self.state)

        if self.algorithm == 'sarsa':
            # Sarsa: 이미 선택된 다음 행동 사용
            action = self._sarsa_next_action if self._sarsa_next_action is not None else self.agent.get_action(self.state, valid_actions)
        else:
            action = self.agent.get_action(self.state, valid_actions)

        next_state, reward, done = self.env.step(action)
        self.step_count += 1

        next_valid_actions = self.env.get_valid_actions(next_state)

        if self.algorithm == 'sarsa':
            if not done:
                next_action = self.agent.get_action(next_state, next_valid_actions)
            else:
                next_action = None
            self.agent.update_q(self.state, action, reward, next_state, next_valid_actions, next_action=next_action)
            self._sarsa_next_action = next_action
        else:
            self.agent.update_q(self.state, action, reward, next_state, next_valid_actions)

        self.state = next_state

        if done:
            self.episode_count += 1
            # 목표 도달인 경우만 성공 카운트
            if self.state == self.env.goal_pos:
                self.success_count += 1
            self.history_steps.append(self.step_count)
            if self.step_count < self.best_steps:
                self.best_steps = self.step_count
            self.spawn_celebration_particles()
            
            # 로그 기록
            success_rate = self.success_count / self.episode_count
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(f"{self.episode_count},{self.step_count},{success_rate:.4f},{self.agent.alpha},{self.agent.gamma},{self.agent.epsilon:.4f},{self.algorithm},{self.env.height}x{self.env.width}\n")
            
            self.agent.decay_epsilon()
            self.state = self.env.reset()
            self.step_count = 0
            # Sarsa: 새 에피소드 시작 시 첫 행동 미리 선택
            if self.algorithm == 'sarsa':
                v = self.env.get_valid_actions(self.state)
                self._sarsa_next_action = self.agent.get_action(self.state, v)
        elif self.step_count >= 300:
            self.episode_count += 1
            self.history_steps.append(self.step_count)
            
            # 로그 기록
            success_rate = self.success_count / self.episode_count
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(f"{self.episode_count},{self.step_count},{success_rate:.4f},{self.agent.alpha},{self.agent.gamma},{self.agent.epsilon:.4f},{self.algorithm},{self.env.height}x{self.env.width}\n")
            
            self.agent.decay_epsilon()
            self.state = self.env.reset()
            self.step_count = 0
            if self.algorithm == 'sarsa':
                v = self.env.get_valid_actions(self.state)
                self._sarsa_next_action = self.agent.get_action(self.state, v)

    def run_fast_training_loop(self, target_episodes):
        """시각화 없이 빠르게 학습합니다."""
        for ep in range(target_episodes):
            state = self.env.reset()
            step = 0

            if self.algorithm == 'sarsa':
                valid_actions = self.env.get_valid_actions(state)
                action = self.agent.get_action(state, valid_actions)

            while True:
                if self.algorithm != 'sarsa':
                    valid_actions = self.env.get_valid_actions(state)
                    action = self.agent.get_action(state, valid_actions)

                next_state, reward, done = self.env.step(action)
                step += 1

                next_valid_actions = self.env.get_valid_actions(next_state)

                if self.algorithm == 'sarsa':
                    next_action = self.agent.get_action(next_state, next_valid_actions) if not done else None
                    self.agent.update_q(state, action, reward, next_state, next_valid_actions, next_action=next_action)
                    action = next_action
                else:
                    self.agent.update_q(state, action, reward, next_state, next_valid_actions)

                state = next_state
                if done:
                    self.episode_count += 1
                    if state == self.env.goal_pos:
                        self.success_count += 1
                    self.history_steps.append(step)
                    if step < self.best_steps:
                        self.best_steps = step
                    
                    # 로그 기록
                    success_rate = self.success_count / self.episode_count
                    with open(self.log_filepath, "a", encoding="utf-8") as f:
                        f.write(f"{self.episode_count},{step},{success_rate:.4f},{self.agent.alpha},{self.agent.gamma},{self.agent.epsilon:.4f},{self.algorithm},{self.env.height}x{self.env.width}\n")
                    
                    self.agent.decay_epsilon()
                    break
                elif step >= 300:
                    self.episode_count += 1
                    self.history_steps.append(step)
                    
                    # 로그 기록
                    success_rate = self.success_count / self.episode_count
                    with open(self.log_filepath, "a", encoding="utf-8") as f:
                        f.write(f"{self.episode_count},{step},{success_rate:.4f},{self.agent.alpha},{self.agent.gamma},{self.agent.epsilon:.4f},{self.algorithm},{self.env.height}x{self.env.width}\n")
                    
                    self.agent.decay_epsilon()
                    break

            self.update_agent_params()

        self.spawn_celebration_particles()
        self.state = self.env.reset()
        self.step_count = 0

    def step_testing(self):
        """학습된 최적 정책으로만 미로를 이동합니다."""
        x, y = self.state
        valid_actions = self.env.get_valid_actions(self.state)

        q_values = [self.agent.q_table[x, y, a] for a in valid_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        action = best_actions[0] if best_actions else 0

        next_state, _, done = self.env.step(action)
        self.step_count += 1
        self.state = next_state

        if done:
            self.spawn_celebration_particles()
            
            # 테스트 결과 로그 기록
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(f"TEST_SUCCESS,{self.step_count},1.0000,0.0,0.0,0.0000,TEST_{self.algorithm},{self.env.height}x{self.env.width}\n")
            
            self.current_state = MazeApp.STATE_IDLE
            self.state = self.env.reset()
            self.step_count = 0
        elif self.step_count >= 300:
            # 테스트 결과 로그 기록
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(f"TEST_FAIL,{self.step_count},0.0000,0.0,0.0,0.0000,TEST_{self.algorithm},{self.env.height}x{self.env.width}\n")
            
            self.current_state = MazeApp.STATE_IDLE
            self.state = self.env.reset()
            self.step_count = 0

    # ──────────────────────────────────────────
    # 에디터 입력 처리
    # ──────────────────────────────────────────
    def handle_editor_click(self, mouse_pos, button):
        """편집 모드에서 미로 격자 클릭 처리"""
        if not self.maze_rect.collidepoint(mouse_pos):
            return

        col = (mouse_pos[0] - self.maze_rect.x) // self.cell_w
        row = (mouse_pos[1] - self.maze_rect.y) // self.cell_h

        if not (0 <= row < self.env.height and 0 <= col < self.env.width):
            return

        if button == 3:  # 우클릭 = 지우기 (길로 변환)
            if (row, col) != self.env.start_pos and (row, col) != self.env.goal_pos:
                self.env.original_map[row, col] = 0
        elif button == 1:  # 좌클릭 = 브러시 적용
            brush = self.editor_brush
            if brush == 2:  # 시작점 이동
                # 기존 시작점을 길로 변환
                sr, sc = self.env.start_pos
                self.env.original_map[sr, sc] = 0
                self.env.original_map[row, col] = 2
                self.env.start_pos = (row, col)
            elif brush == 3:  # 목적지 이동
                gr, gc = self.env.goal_pos
                self.env.original_map[gr, gc] = 0
                self.env.original_map[row, col] = 3
                self.env.goal_pos = (row, col)
            else:
                # 시작점/목적지 위에는 덮어쓰기 금지
                if (row, col) != self.env.start_pos and (row, col) != self.env.goal_pos:
                    self.env.original_map[row, col] = brush

        # 보너스 목록 갱신
        self.env._parse_map()
        self.state = self.env.reset()

    # ──────────────────────────────────────────
    # 그리기(Drawing) 관련 메서드
    # ──────────────────────────────────────────
    def draw_maze(self):
        for r in range(self.env.height):
            for c in range(self.env.width):
                cell_type = self.env.original_map[r, c]
                cell_rect = pygame.Rect(
                    self.maze_rect.x + c * self.cell_w,
                    self.maze_rect.y + r * self.cell_h,
                    self.cell_w,
                    self.cell_h
                )

                # 기본 색상
                if cell_type == 1:
                    pygame.draw.rect(self.screen, COLOR_WALL, cell_rect)
                else:
                    pygame.draw.rect(self.screen, COLOR_PATH, cell_rect)
                    # Q-value 오버레이
                    if self.show_q_overlay and (r, c) != self.env.goal_pos and self.current_state != MazeApp.STATE_EDITING:
                        self.draw_q_overlay(r, c, cell_rect)

                pygame.draw.rect(self.screen, COLOR_GRID_LINE, cell_rect, 1)

                # 특수 타일 장식
                inner_margin = max(2, self.cell_w // 8)
                inner = cell_rect.inflate(-inner_margin * 2, -inner_margin * 2)
                border_rad = max(2, self.cell_w // 10)
                if (r, c) == self.env.start_pos:
                    pygame.draw.rect(self.screen, COLOR_START, inner, border_radius=border_rad)
                elif (r, c) == self.env.goal_pos:
                    pygame.draw.rect(self.screen, COLOR_GOAL, inner, border_radius=border_rad)
                elif cell_type == 4:  # 함정
                    color = COLOR_TRAP if self.env.trap_enabled else COLOR_TRAP_OFF
                    pygame.draw.rect(self.screen, color, inner, border_radius=border_rad)
                    # X 표시
                    x1, y1 = inner.topleft
                    x2, y2 = inner.bottomright
                    line_thick = max(1, self.cell_w // 16)
                    pygame.draw.line(self.screen, (255, 200, 200), (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), line_thick)
                    pygame.draw.line(self.screen, (255, 200, 200), (x2 - 2, y1 + 2), (x1 + 2, y2 - 2), line_thick)
                elif cell_type == 5:  # 보너스
                    if (r, c) in self.env.active_bonuses:
                        pygame.draw.rect(self.screen, COLOR_BONUS, inner, border_radius=border_rad)
                        # ★ 표시
                        cx, cy = inner.center
                        self._draw_star(cx, cy, max(3, inner.width // 3))
                    else:
                        pygame.draw.rect(self.screen, COLOR_BONUS_OFF, inner, border_radius=border_rad)

        # 에이전트 그리기
        if self.current_state != MazeApp.STATE_EDITING:
            agent_r, agent_c = self.state
            cx = self.maze_rect.x + agent_c * self.cell_w + self.cell_w // 2
            cy = self.maze_rect.y + agent_r * self.cell_h + self.cell_h // 2
            r = max(4, self.cell_w // 3)
            pygame.draw.circle(self.screen, COLOR_AGENT, (cx, cy), r)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), max(2, r // 2))

    def _draw_star(self, cx, cy, r):
        """작은 별 모양 그리기"""
        import math
        points = []
        for i in range(5):
            angle = math.radians(-90 + i * 72)
            points.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
            angle2 = math.radians(-90 + i * 72 + 36)
            points.append((cx + int(r * 0.4 * math.cos(angle2)), cy + int(r * 0.4 * math.sin(angle2))))
        pygame.draw.polygon(self.screen, (255, 255, 220), points)

    def draw_q_overlay(self, r, c, rect):
        """격자 칸 내부에 Q-값을 작은 삼각 화살표로 표현합니다."""
        q_vals = self.agent.q_table[r, c]
        max_val = np.max(np.abs(q_vals))
        if max_val < 1e-4:
            return

        directions = [
            (0, -1),  # 상
            (0, 1),   # 하
            (-1, 0),  # 좌
            (1, 0)    # 우
        ]

        for idx, (dx, dy) in enumerate(directions):
            q_val = q_vals[idx]
            if q_val <= 0:
                continue

            ratio = min(1.0, q_val / 1000.0)
            alpha_val = int(50 + 205 * ratio)
            color = (0, 200, 255)

            cx, cy = rect.center
            sz = int(8 * ratio) + 2
            if idx == 0:
                pts = [(cx, cy - sz), (cx - sz, cy), (cx + sz, cy)]
            elif idx == 1:
                pts = [(cx, cy + sz), (cx - sz, cy), (cx + sz, cy)]
            elif idx == 2:
                pts = [(cx - sz, cy), (cx, cy - sz), (cx, cy + sz)]
            else:
                pts = [(cx + sz, cy), (cx, cy - sz), (cx, cy + sz)]

            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            local_pts = [(p[0] - rect.x, p[1] - rect.y) for p in pts]
            pygame.draw.polygon(surf, (*color, alpha_val), local_pts)
            self.screen.blit(surf, rect.topleft)

    # ──────────────────────────────────────────
    # 사이드바 및 그래프
    # ──────────────────────────────────────────
    def draw_sidebar(self):
        sx = 640

        # 사이드바 배경
        sidebar_rect = pygame.Rect(sx - 10, 0, self.screen_w - sx + 10, self.screen_h)
        pygame.draw.rect(self.screen, COLOR_SIDEBAR, sidebar_rect)
        pygame.draw.line(self.screen, COLOR_GRID_LINE, (sx - 10, 0), (sx - 10, self.screen_h), 2)

        # 타이틀
        algo_names = {'q_learning': 'Q-Learning', 'sarsa': 'SARSA', 'double_q': 'Double Q-Learning'}
        title = f"RL 대시보드 — {algo_names.get(self.algorithm, self.algorithm)}"
        title_surf = self.font_title.render(title, True, COLOR_TEXT_MAIN)
        self.screen.blit(title_surf, (sx, 20))

        # 상태 메시지
        state_labels = {
            MazeApp.STATE_IDLE: "대기 중 (Idle)",
            MazeApp.STATE_TRAINING: "학습 진행 중 (Training...)",
            MazeApp.STATE_FAST_TRAINING: "고속 학습 중 (Fast Training...)",
            MazeApp.STATE_TESTING: "최적 경로 테스트 중 (Testing...)",
            MazeApp.STATE_EDITING: "✏️ 편집 모드 (클릭으로 타일 배치)",
        }
        active_states = (MazeApp.STATE_TRAINING, MazeApp.STATE_FAST_TRAINING, MazeApp.STATE_EDITING)
        state_color = COLOR_START if self.current_state in active_states else COLOR_TEXT_MUTED
        state_surf = self.font.render(f"상태: {state_labels.get(self.current_state, '?')}", True, state_color)
        self.screen.blit(state_surf, (sx, 50))

        # 분리선
        pygame.draw.line(self.screen, (60, 60, 75), (sx, 75), (self.screen_w - 20, 75), 1)

        # 알고리즘 선택 라벨
        algo_label = self.font_small.render("알고리즘 선택:", True, COLOR_TEXT_MUTED)
        self.screen.blit(algo_label, (sx, 85))

        # 슬라이더 위 라벨
        param_label = self.font_small.render("하이퍼파라미터:", True, COLOR_TEXT_MUTED)
        self.screen.blit(param_label, (sx, 175))

        # 메트릭 영역 (그래프 위쪽)
        metric_y = 640
        metrics = [
            ("에피소드", f"{self.episode_count}"),
            ("스텝", f"{self.step_count}"),
            ("성공", f"{self.success_count}"),
            ("최단", f"{int(self.best_steps) if self.best_steps != float('inf') else '-'}"),
            ("ε", f"{self.agent.epsilon:.4f}"),
        ]

        # 메트릭을 가로로 나란히 배치
        mx = sx
        for label, val in metrics:
            lbl_surf = self.font_small.render(label, True, COLOR_TEXT_MUTED)
            val_surf = self.font_small.render(val, True, COLOR_TEXT_MAIN)
            self.screen.blit(lbl_surf, (mx, metric_y))
            self.screen.blit(val_surf, (mx, metric_y + 16))
            mx += 70

        # 편집 모드 브러시 레이블
        if self.current_state == MazeApp.STATE_EDITING:
            brush_names = {0: "길", 1: "벽", 4: "함정", 5: "보너스", 2: "시작", 3: "목적지"}
            brush_label = self.font.render(f"브러시: {brush_names.get(self.editor_brush, '?')} | 우클릭=지우기", True, COLOR_AGENT)
            self.screen.blit(brush_label, (sx, 620))

        # 단축키 안내
        tip_surf = self.font_small.render("[Q] Q-값 표시 전환", True, COLOR_TEXT_MUTED)
        self.screen.blit(tip_surf, (sx, self.screen_h - 22))

    def draw_graph(self):
        """사이드바 하단에 실시간 학습 곡선 그래프를 그립니다."""
        sx = 650
        graph_rect = pygame.Rect(sx, 680, 350, 80)

        # 그래프 배경
        pygame.draw.rect(self.screen, (25, 25, 32), graph_rect, border_radius=6)
        pygame.draw.rect(self.screen, (55, 55, 70), graph_rect, width=1, border_radius=6)

        # 제목
        title = self.font_small.render("학습 곡선 (에피소드별 Steps)", True, COLOR_TEXT_MUTED)
        self.screen.blit(title, (sx + 5, graph_rect.y + 3))

        if len(self.history_steps) < 2:
            hint = self.font_small.render("데이터 부족...", True, (80, 80, 100))
            self.screen.blit(hint, (graph_rect.centerx - 30, graph_rect.centery))
            return

        # 최근 100개 데이터만 표시
        data = self.history_steps[-100:]
        max_steps = max(data)
        min_steps = min(data)
        spread = max(max_steps - min_steps, 1)

        # 그래프 내부 여백
        pad_top = 18
        pad_bottom = 4
        pad_left = 5
        pad_right = 5
        gw = graph_rect.width - pad_left - pad_right
        gh = graph_rect.height - pad_top - pad_bottom

        points = []
        for i, s in enumerate(data):
            px = graph_rect.x + pad_left + int(gw * i / (len(data) - 1))
            py = graph_rect.y + pad_top + gh - int(gh * (s - min_steps) / spread)
            points.append((px, py))

        # 꺾은선 그리기
        if len(points) >= 2:
            pygame.draw.lines(self.screen, COLOR_START, False, points, 2)

        # 최소/최대 레이블
        max_label = self.font_small.render(f"{max_steps}", True, COLOR_TEXT_MUTED)
        min_label = self.font_small.render(f"{min_steps}", True, COLOR_TEXT_MUTED)
        self.screen.blit(max_label, (graph_rect.right - max_label.get_width() - 4, graph_rect.y + pad_top - 2))
        self.screen.blit(min_label, (graph_rect.right - min_label.get_width() - 4, graph_rect.bottom - pad_bottom - 12))

    # ──────────────────────────────────────────
    # 메인 루프
    # ──────────────────────────────────────────
    def run(self):
        running = True
        mouse_held = False
        while running:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # 버튼 호버 및 이벤트
                for btn in self.buttons:
                    btn.check_hover(mouse_pos)
                    btn.handle_event(event)
                for btn in self.algo_buttons:
                    btn.check_hover(mouse_pos)
                    btn.handle_event(event)
                for btn in self.edit_buttons:
                    btn.check_hover(mouse_pos)
                    btn.handle_event(event)
                self.trap_toggle_btn.check_hover(mouse_pos)
                self.trap_toggle_btn.handle_event(event)

                # 편집 모드 브러시 버튼
                if self.current_state == MazeApp.STATE_EDITING:
                    for btn in self.brush_buttons:
                        btn.check_hover(mouse_pos)
                        btn.handle_event(event)

                # 슬라이더
                for slider in self.sliders:
                    if slider.handle_event(event, mouse_pos):
                        self.update_agent_params()

                # 편집 모드 마우스 클릭/드래그
                if self.current_state == MazeApp.STATE_EDITING:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                        mouse_held = True
                        self.handle_editor_click(mouse_pos, event.button)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        mouse_held = False
                    elif event.type == pygame.MOUSEMOTION and mouse_held:
                        btn = 1 if pygame.mouse.get_pressed()[0] else 3
                        self.handle_editor_click(mouse_pos, btn)

                # 키보드 이벤트
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.show_q_overlay = not self.show_q_overlay

            # 상태별 업데이트
            if self.current_state == MazeApp.STATE_TRAINING:
                fps_limit = int(self.sliders[2].val)
                self.step_training()
                self.clock.tick(fps_limit)
            elif self.current_state == MazeApp.STATE_TESTING:
                self.step_testing()
                self.clock.tick(10)
            else:
                self.clock.tick(30)

            # 파티클 갱신
            self.update_particles()

            # 화면 렌더링
            self.screen.fill(COLOR_BG)

            self.draw_maze()
            self.draw_sidebar()
            self.draw_graph()
            self.draw_particles()

            # 버튼 그리기
            for btn in self.buttons:
                btn.draw(self.screen, self.font)
            for btn in self.algo_buttons:
                btn.draw(self.screen, self.font)
            for btn in self.edit_buttons:
                btn.draw(self.screen, self.font)
            self.trap_toggle_btn.draw(self.screen, self.font)
            for slider in self.sliders:
                slider.draw(self.screen, self.font)

            # 편집 모드일 때만 브러시 버튼 표시
            if self.current_state == MazeApp.STATE_EDITING:
                for btn in self.brush_buttons:
                    btn.draw(self.screen, self.font)

            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = MazeApp()
    app.run()

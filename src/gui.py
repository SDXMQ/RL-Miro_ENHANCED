import pygame
import numpy as np
import sys
import time
from environment import MazeEnv
from agent import QLearningAgent

# 색상 테마 정의 (HEX -> RGB)
COLOR_BG = (30, 30, 36)          # #1e1e24 - 딥 차콜
COLOR_SIDEBAR = (42, 42, 53)     # #2a2a35 - 다크 블루그레이
COLOR_WALL = (63, 75, 92)        # #3f4b5c - 스틸 블루
COLOR_PATH = (18, 18, 20)        # #121214 - 다크 블랙(길)
COLOR_START = (6, 214, 160)      # #06d6a0 - 네온 민트 (시작)
COLOR_GOAL = (255, 93, 115)      # #ff5d73 - 네온 핑크 (도착)
COLOR_AGENT = (255, 209, 102)    # #ffd166 - 네온 골드 (에이전트)
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
                # 소소한 클릭 효과음 대신 화면을 살짝 갱신하기 위해 이벤트 트리거
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

class MazeApp:
    # 프로그램 실행 상태 상수
    STATE_IDLE = 0
    STATE_TRAINING = 1
    STATE_FAST_TRAINING = 2
    STATE_TESTING = 3

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("RL Maze Solver - Enhanced")
        
        self.screen_w = 1020
        self.screen_h = 670
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.clock = pygame.time.Clock()
        
        # 폰트 초기화 (한글 지원을 위해 기본 시스템 폰트 로드 시도)
        font_candidates = ["malgungothic", "applesdgothicneo", "nanumgothic", "arial", "helvetica"]
        self.font = None
        for fc in font_candidates:
            try:
                self.font = pygame.font.SysFont(fc, 16)
                self.font_title = pygame.font.SysFont(fc, 22, bold=True)
                self.font_big = pygame.font.SysFont(fc, 32, bold=True)
                break
            except Exception:
                continue
        if self.font is None:
            self.font = pygame.font.Font(None, 24)
            self.font_title = pygame.font.Font(None, 32)
            self.font_big = pygame.font.Font(None, 44)

        # 환경 및 에이전트 초기화
        self.env = MazeEnv()
        self.agent = QLearningAgent(
            state_space_shape=(self.env.height, self.env.width),
            alpha=0.15,
            gamma=0.85,
            epsilon=0.999
        )
        
        self.state = self.env.reset()
        self.current_state = MazeApp.STATE_IDLE
        
        # 메트릭 관리
        self.episode_count = 0
        self.step_count = 0
        self.success_count = 0
        self.best_steps = float('inf')
        self.history_steps = []  # 최근 에피소드별 이동 횟수

        # 미로 렌더링 영역 크기 설정 (마진 제외)
        self.maze_rect = pygame.Rect(20, 20, 600, 600)
        self.cell_w = self.maze_rect.width // self.env.width
        self.cell_h = self.maze_rect.height // self.env.height

        # UI 요소 생성
        self.setup_ui()
        
        # 파티클 효과용 목록
        self.particles = []
        
        # Q-value 시각화 스위치
        self.show_q_overlay = True

    def setup_ui(self):
        sidebar_x = 640
        self.buttons = [
            Button(sidebar_x, 320, 170, 40, "학습 시작", BTN_SUCCESS, BTN_SUCCESS_HOVER, action=self.start_training),
            Button(sidebar_x + 180, 320, 170, 40, "일시 정지", BTN_COLOR, BTN_HOVER, action=self.pause_training),
            Button(sidebar_x, 370, 170, 40, "고속 학습 (100회)", BTN_COLOR, BTN_HOVER, action=self.start_fast_training),
            Button(sidebar_x + 180, 370, 170, 40, "테스트 실행", BTN_COLOR, BTN_HOVER, action=self.start_testing),
            Button(sidebar_x, 420, 350, 40, "에이전트 및 Q-Table 초기화", BTN_DANGER, BTN_DANGER_HOVER, action=self.reset_all),
        ]

        # 슬라이더 생성 (x, y, w, h, min, max, start_val, label, is_float)
        self.sliders = [
            Slider(sidebar_x, 150, 350, 10, 0.01, 1.0, self.agent.alpha, "학습률 (Alpha)", is_float=True),
            Slider(sidebar_x, 210, 350, 10, 0.0, 1.0, self.agent.gamma, "감가율 (Gamma)", is_float=True),
            Slider(sidebar_x, 270, 350, 10, 1, 100, 30, "학습 속도 (FPS)"),
        ]

    # --- UI 버튼 액션 핸들러 ---
    def start_training(self):
        if self.current_state in (MazeApp.STATE_IDLE, MazeApp.STATE_TESTING):
            self.current_state = MazeApp.STATE_TRAINING
            # 만약 직전에 테스트 중이거나 완료 상태였다면 리셋 후 시작
            if self.env.current_pos == self.env.goal_pos:
                self.state = self.env.reset()
                self.step_count = 0

    def pause_training(self):
        if self.current_state in (MazeApp.STATE_TRAINING, MazeApp.STATE_FAST_TRAINING):
            self.current_state = MazeApp.STATE_IDLE

    def start_fast_training(self):
        self.current_state = MazeApp.STATE_FAST_TRAINING
        # 고속 학습 실행 (100 에피소드만큼 내부 루프 진행)
        self.run_fast_training_loop(100)
        self.current_state = MazeApp.STATE_IDLE

    def start_testing(self):
        self.current_state = MazeApp.STATE_TESTING
        self.state = self.env.reset()
        self.step_count = 0

    def reset_all(self):
        self.agent = QLearningAgent(
            state_space_shape=(self.env.height, self.env.width),
            alpha=self.sliders[0].val,
            gamma=self.sliders[1].val,
            epsilon=0.999
        )
        self.state = self.env.reset()
        self.current_state = MazeApp.STATE_IDLE
        self.episode_count = 0
        self.step_count = 0
        self.success_count = 0
        self.best_steps = float('inf')
        self.history_steps = []
        self.particles.clear()

    # --- 실시간 학습 및 실행 로직 ---
    def update_agent_params(self):
        """슬라이더 설정 값을 에이전트에 반영합니다."""
        self.agent.alpha = self.sliders[0].val
        self.agent.gamma = self.sliders[1].val

    def spawn_celebration_particles(self):
        # 목적지 도착 시 축하 파티클 생성
        goal_x = self.maze_rect.x + self.env.goal_pos[1] * self.cell_w + self.cell_w // 2
        goal_y = self.maze_rect.y + self.env.goal_pos[0] * self.cell_h + self.cell_h // 2
        for _ in range(30):
            import random
            color = random.choice([COLOR_START, COLOR_AGENT, (255, 100, 200)])
            speed_x = random.uniform(-4, 4)
            speed_y = random.uniform(-6, -1)
            life = random.randint(30, 60)
            size = random.randint(3, 7)
            self.particles.append([goal_x, goal_y, speed_x, speed_y, life, color, size])

    def update_particles(self):
        for p in self.particles[:]:
            p[0] += p[2]  # x += speed_x
            p[1] += p[3]  # y += speed_y
            p[3] += 0.15  # 중력 효과
            p[4] -= 1     # 수명 감소
            if p[4] <= 0:
                self.particles.remove(p)

    def draw_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), p[6])

    def step_training(self):
        """1단계 학습 업데이트를 수행합니다."""
        # 1. 행동 선택
        valid_actions = self.env.get_valid_actions(self.state)
        action = self.agent.get_action(self.state, valid_actions)
        
        # 2. 환경 한 단계 진행
        next_state, reward, done = self.env.step(action)
        self.step_count += 1
        
        # 3. 에이전트 Q-Table 업데이트
        next_valid_actions = self.env.get_valid_actions(next_state)
        self.agent.update_q(self.state, action, reward, next_state, next_valid_actions)
        
        # 4. 상태 전이
        self.state = next_state
        
        # 5. 에피소드 종료 판정
        if done:
            self.episode_count += 1
            self.success_count += 1
            self.history_steps.append(self.step_count)
            if self.step_count < self.best_steps:
                self.best_steps = self.step_count
            
            self.spawn_celebration_particles()
            self.agent.decay_epsilon()
            
            # 리셋
            self.state = self.env.reset()
            self.step_count = 0
        elif self.step_count >= 150: # 한 에피소드 최대 스텝 수 제한
            self.episode_count += 1
            self.agent.decay_epsilon()
            self.state = self.env.reset()
            self.step_count = 0

    def run_fast_training_loop(self, target_episodes):
        """시각화 없이 빠르게 Q-러닝을 실행합니다."""
        for ep in range(target_episodes):
            state = self.env.reset()
            step = 0
            while True:
                valid_actions = self.env.get_valid_actions(state)
                action = self.agent.get_action(state, valid_actions)
                next_state, reward, done = self.env.step(action)
                step += 1
                
                next_valid_actions = self.env.get_valid_actions(next_state)
                self.agent.update_q(state, action, reward, next_state, next_valid_actions)
                
                state = next_state
                if done:
                    self.episode_count += 1
                    self.success_count += 1
                    self.history_steps.append(step)
                    if step < self.best_steps:
                        self.best_steps = step
                    self.agent.decay_epsilon()
                    break
                elif step >= 150:
                    self.episode_count += 1
                    self.agent.decay_epsilon()
                    break
            
            # 실시간 파라미터 업데이트 수동 적용
            self.update_agent_params()

        self.spawn_celebration_particles()
        self.state = self.env.reset()
        self.step_count = 0

    def step_testing(self):
        """학습된 최적 정책으로만 미로를 이동합니다."""
        x, y = self.state
        valid_actions = self.env.get_valid_actions(self.state)
        
        # 완전 탐욕(Exploitation) 행동 결정
        q_values = [self.agent.q_table[x, y, a] for a in valid_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        action = best_actions[0] if best_actions else 0
        
        next_state, _, done = self.env.step(action)
        self.step_count += 1
        self.state = next_state
        
        if done:
            self.spawn_celebration_particles()
            self.current_state = MazeApp.STATE_IDLE
            self.state = self.env.reset()
        elif self.step_count >= 150:
            self.current_state = MazeApp.STATE_IDLE
            self.state = self.env.reset()

    # --- 그리기(Drawing) 관련 메서드 ---
    def draw_maze(self):
        # 미로 격자 그리기
        for r in range(self.env.height):
            for c in range(self.env.width):
                cell_type = self.env.original_map[r, c]
                cell_rect = pygame.Rect(
                    self.maze_rect.x + c * self.cell_w,
                    self.maze_rect.y + r * self.cell_h,
                    self.cell_w,
                    self.cell_h
                )
                
                # 색상 채우기
                if cell_type == 1:
                    pygame.draw.rect(self.screen, COLOR_WALL, cell_rect)
                    pygame.draw.rect(self.screen, COLOR_GRID_LINE, cell_rect, 1)
                else:
                    pygame.draw.rect(self.screen, COLOR_PATH, cell_rect)
                    pygame.draw.rect(self.screen, COLOR_GRID_LINE, cell_rect, 1)
                    
                    # Q-value 오버레이 시각화
                    if self.show_q_overlay and (r, c) != self.env.goal_pos:
                        self.draw_q_overlay(r, c, cell_rect)
                
                # 시작/목적지 표식 그리기 (둥글고 은은하게)
                if (r, c) == self.env.start_pos:
                    pygame.draw.rect(self.screen, COLOR_START, cell_rect.inflate(-8, -8), border_radius=6)
                elif (r, c) == self.env.goal_pos:
                    pygame.draw.rect(self.screen, COLOR_GOAL, cell_rect.inflate(-8, -8), border_radius=6)

        # 에이전트(☆) 그리기
        agent_r, agent_c = self.state
        agent_rect = pygame.Rect(
            self.maze_rect.x + agent_c * self.cell_w,
            self.maze_rect.y + agent_r * self.cell_h,
            self.cell_w,
            self.cell_h
        ).inflate(-12, -12)
        pygame.draw.circle(self.screen, COLOR_AGENT, agent_rect.center, agent_rect.width // 2)
        # 에이전트 내부에 작은 포인트 주기
        pygame.draw.circle(self.screen, (255, 255, 255), agent_rect.center, agent_rect.width // 4)

    def draw_q_overlay(self, r, c, rect):
        """격자 칸 내부에 Q-값을 작은 삼각 화살표로 표현합니다."""
        # 4방향: 상, 하, 좌, 우
        q_vals = self.agent.q_table[r, c]
        max_val = np.max(np.abs(q_vals))
        if max_val < 1e-4:
            return  # Q값이 거의 없으면 그리지 않음

        # 방향 벡터 정의
        directions = [
            (0, -1),  # 상 (화면 기준 y가 위쪽으로 감소해야 함)
            (0, 1),   # 하
            (-1, 0),  # 좌
            (1, 0)    # 우
        ]
        
        for idx, (dx, dy) in enumerate(directions):
            q_val = q_vals[idx]
            if q_val <= 0:
                continue # 양수 보상인 방향에만 강하게 표시
                
            # 화살표 크기 조절
            ratio = min(1.0, q_val / 1000.0) # 최대 보상 1000 기준
            alpha_val = int(50 + 205 * ratio)
            color = (0, 200, 255) # 청량한 하늘색 화살표
            
            # 삼각형 그리기
            cx, cy = rect.center
            sz = int(8 * ratio) + 2
            if idx == 0: # 상
                pts = [(cx, cy - sz), (cx - sz, cy), (cx + sz, cy)]
            elif idx == 1: # 하
                pts = [(cx, cy + sz), (cx - sz, cy), (cx + sz, cy)]
            elif idx == 2: # 좌
                pts = [(cx - sz, cy), (cx, cy - sz), (cx, cy + sz)]
            else: # 우
                pts = [(cx + sz, cy), (cx, cy - sz), (cx, cy + sz)]
            
            # pygame 윈도우 투명도 처리를 위해 표면에 그려 서페이서 조합
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            local_pts = [(p[0] - rect.x, p[1] - rect.y) for p in pts]
            pygame.draw.polygon(surf, (*color, alpha_val), local_pts)
            self.screen.blit(surf, rect.topleft)

    def draw_sidebar(self):
        sidebar_x = 640
        # 사이드바 배경 채우기
        sidebar_rect = pygame.Rect(sidebar_x - 10, 0, self.screen_w - sidebar_x + 10, self.screen_h)
        pygame.draw.rect(self.screen, COLOR_SIDEBAR, sidebar_rect)
        pygame.draw.line(self.screen, COLOR_GRID_LINE, (sidebar_x - 10, 0), (sidebar_x - 10, self.screen_h), 2)
        
        # 타이틀
        title_surf = self.font_title.render("Q-Learning 대시보드", True, COLOR_TEXT_MAIN)
        self.screen.blit(title_surf, (sidebar_x, 25))
        
        # 상태 메시지
        state_labels = {
            MazeApp.STATE_IDLE: "대기 중 (Idle)",
            MazeApp.STATE_TRAINING: "학습 진행 중 (Training...)",
            MazeApp.STATE_FAST_TRAINING: "고속 학습 중 (Fast Training...)",
            MazeApp.STATE_TESTING: "최적 경로 테스트 중 (Testing...)"
        }
        state_color = COLOR_START if self.current_state in (MazeApp.STATE_TRAINING, MazeApp.STATE_FAST_TRAINING) else COLOR_TEXT_MUTED
        state_surf = self.font.render(f"상태: {state_labels[self.current_state]}", True, state_color)
        self.screen.blit(state_surf, (sidebar_x, 60))
        
        # 분리선
        pygame.draw.line(self.screen, (60, 60, 75), (sidebar_x, 90), (self.screen_w - 20, 90), 1)

        # 수치 지표 메트릭 영역
        metric_y = 485
        metrics = [
            ("진행 에피소드", f"{self.episode_count} 회차"),
            ("현재 스텝 수", f"{self.step_count} Steps"),
            ("학습 성공 횟수", f"{self.success_count} 회"),
            ("최적 이동 (최단)", f"{self.best_steps if self.best_steps != float('inf') else '-'} Steps"),
            ("탐험률 (Epsilon)", f"{self.agent.epsilon:.4f}"),
        ]
        
        for idx, (label, val) in enumerate(metrics):
            y_pos = metric_y + idx * 26
            lbl_surf = self.font.render(label, True, COLOR_TEXT_MUTED)
            val_surf = self.font.render(val, True, COLOR_TEXT_MAIN)
            self.screen.blit(lbl_surf, (sidebar_x, y_pos))
            self.screen.blit(val_surf, (self.screen_w - 20 - val_surf.get_width(), y_pos))

        # 조작 설명 단축키
        tip_y = 615
        tip_surf = self.font.render("[Q] 키: Q-값 화살표 표시 전환", True, COLOR_TEXT_MUTED)
        self.screen.blit(tip_surf, (sidebar_x, tip_y))

    def run(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            # 이벤트 루프
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # 버튼 호버 상태 체크 및 이벤트 위임
                for btn in self.buttons:
                    btn.check_hover(mouse_pos)
                    btn.handle_event(event)
                
                # 슬라이더 조작 위임
                for slider in self.sliders:
                    if slider.handle_event(event, mouse_pos):
                        self.update_agent_params()
                
                # 키보드 이벤트
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.show_q_overlay = not self.show_q_overlay
            
            # 훈련/테스트 실시간 업데이트
            if self.current_state == MazeApp.STATE_TRAINING:
                # 학습 FPS 제한 (3단계 슬라이더 값 활용)
                fps_limit = int(self.sliders[2].val)
                self.step_training()
                self.clock.tick(fps_limit)
            elif self.current_state == MazeApp.STATE_TESTING:
                # 테스트 시에는 약 10 FPS 수준의 가독성 좋은 속도로 이동
                self.step_testing()
                self.clock.tick(10)
            else:
                self.clock.tick(30) # 대기 중 상태의 부드러운 프레임

            # 파티클 갱신
            self.update_particles()
            
            # 화면 지우기 및 렌더링
            self.screen.fill(COLOR_BG)
            
            self.draw_maze()
            self.draw_sidebar()
            self.draw_particles()
            
            # 버튼 및 슬라이더 그리기
            for btn in self.buttons:
                btn.draw(self.screen, self.font)
            for slider in self.sliders:
                slider.draw(self.screen, self.font)

            pygame.display.flip()
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = MazeApp()
    app.run()

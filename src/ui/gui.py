import numpy as np
import pygame
import sys
import random
from rl.environment import MazeEnv
from rl.agent import RLAgent
from ui.widgets import Button, Slider, ToggleButton
from utils.generator import generate_maze_dfs
from utils.i18n import i18n


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

        # 설정 변수 및 상태 초기화
        self.sound_enabled = True
        self.particles_enabled = True
        self.show_grid = True
        self.theme = 'dark'
        self.show_settings = False
        
        self.update_theme_colors()
        pygame.display.set_caption(i18n.get("window_title"))

        # UI 요소 생성
        self.setup_ui()

    def update_theme_colors(self):
        if self.theme == 'dark':
            self.color_bg = (30, 30, 36)
            self.color_sidebar = (42, 42, 53)
            self.color_wall = (63, 75, 92)
            self.color_path = (18, 18, 20)
            self.color_text_main = (255, 255, 255)
            self.color_text_muted = (160, 160, 180)
            self.color_grid_line = (35, 35, 45)
        elif self.theme == 'classic':
            self.color_bg = (20, 40, 40)
            self.color_sidebar = (25, 55, 55)
            self.color_wall = (100, 120, 120)
            self.color_path = (10, 20, 20)
            self.color_text_main = (240, 255, 240)
            self.color_text_muted = (150, 180, 180)
            self.color_grid_line = (30, 60, 60)
        elif self.theme == 'light':
            self.color_bg = (245, 245, 248)
            self.color_sidebar = (230, 230, 240)
            self.color_wall = (180, 190, 205)
            self.color_path = (255, 255, 255)
            self.color_text_main = (30, 30, 40)
            self.color_text_muted = (100, 100, 120)
            self.color_grid_line = (210, 210, 220)

    def play_beep(self, freq=440, duration=0.1):
        if not self.sound_enabled:
            return
        try:
            import math
            import array
            sample_rate = 22050
            n_samples = int(sample_rate * duration)
            buf = array.array('h', [int(32767 * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n_samples)])
            sound = pygame.mixer.Sound(buffer=buf)
            sound.set_volume(0.2)
            sound.play()
        except Exception:
            pass


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

        # 슬라이더 값 보존
        alpha_val = self.sliders[0].val if hasattr(self, 'sliders') and len(self.sliders) > 0 else self.agent.alpha
        gamma_val = self.sliders[1].val if hasattr(self, 'sliders') and len(self.sliders) > 1 else self.agent.gamma
        fps_val = self.sliders[2].val if hasattr(self, 'sliders') and len(self.sliders) > 2 else 30
        size_val = self.sliders[3].val if hasattr(self, 'sliders') and len(self.sliders) > 3 else self.env.height

        # --- 설정 버튼 (우측 상단 ⚙) ---
        self.settings_btn = Button(self.screen_w - 45, 18, 30, 30, "⚙", (60, 60, 75), (90, 90, 110), action=self.toggle_settings)

        # --- 알고리즘 선택 버튼 (토글) ---
        algo_y = 100
        self.algo_buttons = [
            Button(sx, algo_y, 110, 30, i18n.get('algo_q_learning'), BTN_SUCCESS, BTN_SUCCESS_HOVER, action=lambda: self.set_algorithm('q_learning')),
            Button(sx + 120, algo_y, 110, 30, i18n.get('algo_sarsa'), BTN_COLOR, BTN_HOVER, action=lambda: self.set_algorithm('sarsa')),
            Button(sx + 240, algo_y, 110, 30, i18n.get('algo_double_q'), BTN_PURPLE, BTN_PURPLE_HOVER, action=lambda: self.set_algorithm('double_q')),
        ]

        # --- 함정 토글 버튼 ---
        trap_text = i18n.get('trap_label_on') if self.env.trap_enabled else i18n.get('trap_label_off')
        trap_color = BTN_DANGER if self.env.trap_enabled else (100, 60, 60)
        trap_hover = BTN_DANGER_HOVER if self.env.trap_enabled else (130, 80, 80)
        self.trap_toggle_btn = Button(sx, 140, bw2, 28, trap_text, trap_color, trap_hover, action=self.toggle_trap)

        # --- 슬라이더 ---
        slider_y = 190
        self.sliders = [
            Slider(sx, slider_y, bw2, 10, 0.01, 1.0, alpha_val, i18n.get('slider_alpha'), is_float=True),
            Slider(sx, slider_y + 48, bw2, 10, 0.0, 1.0, gamma_val, i18n.get('slider_gamma'), is_float=True),
            Slider(sx, slider_y + 96, bw2, 10, 1, 100, fps_val, i18n.get('slider_fps')),
            Slider(sx, slider_y + 144, bw2, 10, 5, 30, size_val, i18n.get('slider_size')),
        ]

        # --- 시뮬레이션 제어 버튼 ---
        ctrl_y = 380
        self.buttons = [
            Button(sx, ctrl_y, bw, 34, i18n.get('btn_start'), BTN_SUCCESS, BTN_SUCCESS_HOVER, action=self.start_training),
            Button(sx + 180, ctrl_y, bw, 34, i18n.get('btn_pause'), BTN_COLOR, BTN_HOVER, action=self.pause_training),
            Button(sx, ctrl_y + 42, bw, 34, i18n.get('btn_fast'), BTN_COLOR, BTN_HOVER, action=self.start_fast_training),
            Button(sx + 180, ctrl_y + 42, bw, 34, i18n.get('btn_test'), BTN_COLOR, BTN_HOVER, action=self.start_testing),
            Button(sx, ctrl_y + 84, bw2, 34, i18n.get('btn_reset'), BTN_DANGER, BTN_DANGER_HOVER, action=self.reset_all),
        ]

        # --- 에디터/유틸 버튼 ---
        edit_y = ctrl_y + 130
        self.edit_buttons = [
            Button(sx, edit_y, bw, 34, i18n.get('btn_edit'), BTN_WARN, BTN_WARN_HOVER, action=self.toggle_edit_mode),
            Button(sx + 180, edit_y, bw, 34, i18n.get('btn_random'), BTN_PURPLE, BTN_PURPLE_HOVER, action=self.generate_random_maze),
            Button(sx, edit_y + 42, bw, 34, i18n.get('btn_save'), BTN_COLOR, BTN_HOVER, action=self.save_model),
            Button(sx + 180, edit_y + 42, bw, 34, i18n.get('btn_load'), BTN_COLOR, BTN_HOVER, action=self.load_model),
        ]

        # --- 에디터 브러시 버튼 (편집 모드일 때만 표시) ---
        brush_y = edit_y + 84
        self.brush_buttons = [
            Button(sx, brush_y, 68, 28, i18n.get('brush_wall'), self.color_wall, (90, 100, 120), action=lambda: self._set_brush(1)),
            Button(sx + 72, brush_y, 68, 28, i18n.get('brush_path'), (60, 60, 70), (90, 90, 100), action=lambda: self._set_brush(0)),
            Button(sx + 144, brush_y, 68, 28, i18n.get('brush_trap'), COLOR_TRAP, BTN_DANGER_HOVER, action=lambda: self._set_brush(4)),
            Button(sx + 216, brush_y, 68, 28, i18n.get('brush_bonus'), COLOR_BONUS, (100, 180, 255), action=lambda: self._set_brush(5)),
            Button(sx + 288, brush_y, 68, 28, i18n.get('brush_start'), COLOR_START, (30, 240, 180), action=lambda: self._set_brush(2)),
        ]


    def toggle_settings(self):
        self.show_settings = not self.show_settings
        self.play_beep(523, 0.08)  # C5 음
        if self.show_settings:
            self.setup_settings_ui()

    def setup_settings_ui(self):
        mx = (self.screen_w - 500) // 2
        my = (self.screen_h - 450) // 2

        self.settings_widgets = []
        
        c_inactive = (60, 60, 75)
        c_inactive_hover = (80, 80, 100)
        c_active = BTN_SUCCESS
        c_active_hover = BTN_SUCCESS_HOVER

        # 1. 언어 선택 (ko / en)
        ly = my + 80
        btn_ko = ToggleButton(mx + 200, ly, 100, 28, "한국어", c_inactive, c_inactive_hover, c_active, c_active_hover, 
                              is_active=(i18n.current_lang == 'ko'), action=lambda: self.change_language('ko'))
        btn_en = ToggleButton(mx + 310, ly, 100, 28, "English", c_inactive, c_inactive_hover, c_active, c_active_hover, 
                              is_active=(i18n.current_lang == 'en'), action=lambda: self.change_language('en'))
        self.settings_widgets.extend([btn_ko, btn_en])

        # 2. 테마 선택 (dark / classic / light)
        ty = my + 130
        btn_theme_dark = ToggleButton(mx + 200, ty, 80, 28, i18n.get('theme_dark'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                     is_active=(self.theme == 'dark'), action=lambda: self.change_theme('dark'))
        btn_theme_classic = ToggleButton(mx + 290, ty, 80, 28, i18n.get('theme_classic'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                        is_active=(self.theme == 'classic'), action=lambda: self.change_theme('classic'))
        btn_theme_light = ToggleButton(mx + 380, ty, 80, 28, i18n.get('theme_light'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                      is_active=(self.theme == 'light'), action=lambda: self.change_theme('light'))
        self.settings_widgets.extend([btn_theme_dark, btn_theme_classic, btn_theme_light])

        # 3. 사운드 효과 (ON / OFF)
        sy = my + 180
        btn_sound_on = ToggleButton(mx + 200, sy, 100, 28, i18n.get('state_on'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                    is_active=self.sound_enabled, action=lambda: self.set_sound(True))
        btn_sound_off = ToggleButton(mx + 310, sy, 100, 28, i18n.get('state_off'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                     is_active=not self.sound_enabled, action=lambda: self.set_sound(False))
        self.settings_widgets.extend([btn_sound_on, btn_sound_off])

        # 4. 격자선 표시 (ON / OFF)
        gy = my + 230
        btn_grid_on = ToggleButton(mx + 200, gy, 100, 28, i18n.get('state_on'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                   is_active=self.show_grid, action=lambda: self.set_grid(True))
        btn_grid_off = ToggleButton(mx + 310, gy, 100, 28, i18n.get('state_off'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                    is_active=not self.show_grid, action=lambda: self.set_grid(False))
        self.settings_widgets.extend([btn_grid_on, btn_grid_off])

        # 5. 파티클 효과 (ON / OFF)
        py = my + 280
        btn_part_on = ToggleButton(mx + 200, py, 100, 28, i18n.get('state_on'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                   is_active=self.particles_enabled, action=lambda: self.set_particles(True))
        btn_part_off = ToggleButton(mx + 310, py, 100, 28, i18n.get('state_off'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                    is_active=not self.particles_enabled, action=lambda: self.set_particles(False))
        self.settings_widgets.extend([btn_part_on, btn_part_off])

        # 6. Q-값 표시 (ON / OFF)
        qy = my + 330
        btn_q_on = ToggleButton(mx + 200, qy, 100, 28, i18n.get('state_on'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                 is_active=self.show_q_overlay, action=lambda: self.set_q_overlay(True))
        btn_q_off = ToggleButton(mx + 310, qy, 100, 28, i18n.get('state_off'), c_inactive, c_inactive_hover, c_active, c_active_hover, 
                                  is_active=not self.show_q_overlay, action=lambda: self.set_q_overlay(False))
        self.settings_widgets.extend([btn_q_on, btn_q_off])

        # 7. 닫기 버튼
        btn_close = Button(mx + 175, my + 390, 150, 36, i18n.get('settings_close'), BTN_DANGER, BTN_DANGER_HOVER, action=self.toggle_settings)
        self.settings_widgets.append(btn_close)

    def change_language(self, lang):
        i18n.set_language(lang)
        pygame.display.set_caption(i18n.get("window_title"))
        self.play_beep(659, 0.08)  # E5 음
        self.setup_ui()
        self.setup_settings_ui()

    def change_theme(self, theme):
        self.theme = theme
        self.update_theme_colors()
        self.play_beep(698, 0.08)  # F5 음
        self.setup_ui()
        self.setup_settings_ui()

    def set_sound(self, enabled):
        self.sound_enabled = enabled
        if self.sound_enabled:
            self.play_beep(784, 0.08)  # G5 음
        self.setup_settings_ui()

    def set_grid(self, enabled):
        self.show_grid = enabled
        self.play_beep(784, 0.08)
        self.setup_settings_ui()

    def set_particles(self, enabled):
        self.particles_enabled = enabled
        self.play_beep(784, 0.08)
        self.setup_settings_ui()

    def set_q_overlay(self, enabled):
        self.show_q_overlay = enabled
        self.play_beep(784, 0.08)
        self.setup_settings_ui()

    def draw_settings_modal(self):
        # 1. 반투명 배경 어둡게 오버레이
        overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        # 2. 모달 영역
        mx = (self.screen_w - 500) // 2
        my = (self.screen_h - 450) // 2
        modal_rect = pygame.Rect(mx, my, 500, 450)

        # 그림자 효과
        shadow_rect = pygame.Rect(mx + 4, my + 4, 500, 450)
        shadow_surf = pygame.Surface((500, 450), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (10, 10, 15, 80), (0, 0, 500, 450), border_radius=12)
        self.screen.blit(shadow_surf, shadow_rect.topleft)

        # 테마별 색상 설정
        if self.theme == 'light':
            modal_bg = (240, 240, 245)
            border_color = (180, 180, 195)
            text_color = (30, 30, 45)
        else:
            modal_bg = (35, 35, 48)
            border_color = (75, 75, 95)
            text_color = (240, 240, 250)

        # 모달 본체 그리기
        pygame.draw.rect(self.screen, modal_bg, modal_rect, border_radius=12)
        pygame.draw.rect(self.screen, border_color, modal_rect, width=2, border_radius=12)

        # 타이틀
        title_surf = self.font_title.render(i18n.get('settings_title'), True, text_color)
        title_rect = title_surf.get_rect(centerx=modal_rect.centerx, y=my + 20)
        self.screen.blit(title_surf, title_rect)

        # 구분선
        pygame.draw.line(self.screen, border_color, (mx + 30, my + 55), (mx + 470, my + 55), 1)

        # 항목 라벨
        labels = [
            (i18n.get('settings_lang'), my + 85),
            (i18n.get('settings_theme'), my + 135),
            (i18n.get('settings_sound'), my + 185),
            (i18n.get('settings_grid'), my + 235),
            (i18n.get('settings_particles'), my + 285),
            (i18n.get('settings_q_overlay'), my + 335),
        ]
        for label_text, ly in labels:
            lbl_surf = self.font.render(label_text, True, text_color)
            self.screen.blit(lbl_surf, (mx + 40, ly))

        # 설정 위젯 그리기
        for widget in self.settings_widgets:
            widget.draw(self.screen, self.font)


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
            self.trap_toggle_btn.text = i18n.get('trap_label_on')
            self.trap_toggle_btn.color = BTN_DANGER
            self.trap_toggle_btn.hover_color = BTN_DANGER_HOVER
        else:
            self.trap_toggle_btn.text = i18n.get('trap_label_off')
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
                    self.trap_toggle_btn.text = i18n.get('trap_label_on')
                    self.trap_toggle_btn.color = BTN_DANGER
                    self.trap_toggle_btn.hover_color = BTN_DANGER_HOVER
                else:
                    self.trap_toggle_btn.text = i18n.get('trap_label_off')
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
        if not self.particles_enabled:
            return
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
            self.agent.update_q(self.state, action, reward, next_state, next_valid_actions, done, next_action=next_action)
            self._sarsa_next_action = next_action
        else:
            self.agent.update_q(self.state, action, reward, next_state, next_valid_actions, done)

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
                    self.agent.update_q(state, action, reward, next_state, next_valid_actions, done, next_action=next_action)
                    action = next_action
                else:
                    self.agent.update_q(state, action, reward, next_state, next_valid_actions, done)

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
            self.play_beep(880, 0.15)  # 성공 효과음 (A5)
            
            # 테스트 결과 로그 기록
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(f"TEST_SUCCESS,{self.step_count},1.0000,0.0,0.0,0.0000,TEST_{self.algorithm},{self.env.height}x{self.env.width}\n")
            
            self.current_state = MazeApp.STATE_IDLE
            self.state = self.env.reset()
            self.step_count = 0
        elif self.step_count >= 300:
            self.play_beep(220, 0.25)  # 실패 효과음 (A3)
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

                # 기본 색상 (테마 컬러 적용)
                if cell_type == 1:
                    pygame.draw.rect(self.screen, self.color_wall, cell_rect)
                else:
                    pygame.draw.rect(self.screen, self.color_path, cell_rect)
                    # Q-value 오버레이
                    if self.show_q_overlay and (r, c) != self.env.goal_pos and self.current_state != MazeApp.STATE_EDITING:
                        self.draw_q_overlay(r, c, cell_rect)

                # 격자선 설정 적용
                if self.show_grid:
                    pygame.draw.rect(self.screen, self.color_grid_line, cell_rect, 1)

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

        # 사이드바 배경 (테마 반영)
        sidebar_rect = pygame.Rect(sx - 10, 0, self.screen_w - sx + 10, self.screen_h)
        pygame.draw.rect(self.screen, self.color_sidebar, sidebar_rect)
        pygame.draw.line(self.screen, self.color_grid_line, (sx - 10, 0), (sx - 10, self.screen_h), 2)

        # 타이틀 (i18n 반영)
        algo_names = {
            'q_learning': i18n.get('algo_q_learning'), 
            'sarsa': i18n.get('algo_sarsa'), 
            'double_q': i18n.get('algo_double_q')
        }
        title = f"{i18n.get('dashboard_title')}{algo_names.get(self.algorithm, self.algorithm)}"
        title_surf = self.font_title.render(title, True, self.color_text_main)
        self.screen.blit(title_surf, (sx, 20))

        # 상태 메시지 (i18n 반영)
        state_labels = {
            MazeApp.STATE_IDLE: i18n.get('status_idle'),
            MazeApp.STATE_TRAINING: i18n.get('status_training'),
            MazeApp.STATE_FAST_TRAINING: i18n.get('status_fast_training'),
            MazeApp.STATE_TESTING: i18n.get('status_testing'),
            MazeApp.STATE_EDITING: i18n.get('status_editing'),
        }
        active_states = (MazeApp.STATE_TRAINING, MazeApp.STATE_FAST_TRAINING, MazeApp.STATE_EDITING)
        state_color = COLOR_START if self.current_state in active_states else self.color_text_muted
        state_text = f"{i18n.get('status_label')}{state_labels.get(self.current_state, '?')}"
        state_surf = self.font.render(state_text, True, state_color)
        self.screen.blit(state_surf, (sx, 50))

        # 분리선
        pygame.draw.line(self.screen, self.color_grid_line, (sx, 75), (self.screen_w - 20, 75), 1)

        # 알고리즘 선택 라벨 (i18n 반영)
        algo_label = self.font_small.render(i18n.get('select_algorithm'), True, self.color_text_muted)
        self.screen.blit(algo_label, (sx, 85))

        # 슬라이더 위 라벨 (i18n 반영)
        param_label = self.font_small.render(i18n.get('hyperparameters'), True, self.color_text_muted)
        self.screen.blit(param_label, (sx, 175))

        # 메트릭 영역 (i18n 반영)
        metric_y = 640
        metrics = [
            (i18n.get('metric_episode'), f"{self.episode_count}"),
            (i18n.get('metric_step'), f"{self.step_count}"),
            (i18n.get('metric_success'), f"{self.success_count}"),
            (i18n.get('metric_best'), f"{int(self.best_steps) if self.best_steps != float('inf') else '-'}"),
            (i18n.get('metric_epsilon'), f"{self.agent.epsilon:.4f}"),
        ]

        # 메트릭을 가로로 나란히 배치
        mx = sx
        for label, val in metrics:
            lbl_surf = self.font_small.render(label, True, self.color_text_muted)
            val_surf = self.font_small.render(val, True, self.color_text_main)
            self.screen.blit(lbl_surf, (mx, metric_y))
            self.screen.blit(val_surf, (mx, metric_y + 16))
            mx += 70

        # 편집 모드 브러시 레이블 (i18n 반영)
        if self.current_state == MazeApp.STATE_EDITING:
            brush_names = {
                0: i18n.get('brush_path'), 
                1: i18n.get('brush_wall'), 
                4: i18n.get('brush_trap'), 
                5: i18n.get('brush_bonus'), 
                2: i18n.get('brush_start'), 
                3: i18n.get('brush_goal')
            }
            brush_info_text = i18n.get('brush_info', name=brush_names.get(self.editor_brush, '?'))
            brush_label = self.font.render(brush_info_text, True, COLOR_AGENT)
            self.screen.blit(brush_label, (sx, 620))

        # 단축키 안내 (i18n 반영)
        tip_surf = self.font_small.render(i18n.get('tip_q'), True, self.color_text_muted)
        self.screen.blit(tip_surf, (sx, self.screen_h - 22))


    def draw_graph(self):
        """사이드바 하단에 실시간 학습 곡선 그래프를 그립니다."""
        sx = 650
        graph_rect = pygame.Rect(sx, 680, 350, 80)

        # 그래프 배경 (테마 반영)
        pygame.draw.rect(self.screen, (25, 25, 32), graph_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.color_grid_line, graph_rect, width=1, border_radius=6)

        # 제목 (i18n 반영)
        title = self.font_small.render(i18n.get('graph_title'), True, self.color_text_muted)
        self.screen.blit(title, (sx + 5, graph_rect.y + 3))

        if len(self.history_steps) < 2:
            hint = self.font_small.render(i18n.get('graph_no_data'), True, self.color_text_muted)
            self.screen.blit(hint, (graph_rect.centerx - hint.get_width() // 2, graph_rect.centery))
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

        # 최소/최대 레이블 (테마 반영)
        max_label = self.font_small.render(f"{max_steps}", True, self.color_text_muted)
        min_label = self.font_small.render(f"{min_steps}", True, self.color_text_muted)
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

                # ⚙ 설정 버튼은 항상 작동
                self.settings_btn.check_hover(mouse_pos)
                settings_btn_clicked = self.settings_btn.handle_event(event)

                if self.show_settings:
                    # 설정창 활성화 상태: 설정창 위젯만 상호작용
                    if not settings_btn_clicked:
                        for btn in self.settings_widgets:
                            btn.check_hover(mouse_pos)
                            btn.handle_event(event)
                    continue

                # 버튼 호버 및 이벤트
                for btn in self.buttons:
                    btn.check_hover(mouse_pos)
                    if btn.handle_event(event):
                        self.play_beep(523, 0.05)
                for btn in self.algo_buttons:
                    btn.check_hover(mouse_pos)
                    if btn.handle_event(event):
                        self.play_beep(587, 0.05)
                for btn in self.edit_buttons:
                    btn.check_hover(mouse_pos)
                    if btn.handle_event(event):
                        self.play_beep(523, 0.05)
                self.trap_toggle_btn.check_hover(mouse_pos)
                if self.trap_toggle_btn.handle_event(event):
                    self.play_beep(523, 0.05)

                # 편집 모드 브러시 버튼
                if self.current_state == MazeApp.STATE_EDITING:
                    for btn in self.brush_buttons:
                        btn.check_hover(mouse_pos)
                        if btn.handle_event(event):
                            self.play_beep(659, 0.05)

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
            self.screen.fill(self.color_bg)

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

            # 설정 ⚙ 버튼 그리기
            self.settings_btn.draw(self.screen, self.font)

            # 설정 플로팅창 그리기
            if self.show_settings:
                self.draw_settings_modal()

            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = MazeApp()
    app.run()

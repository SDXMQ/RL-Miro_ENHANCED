class I18NManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(I18NManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.init_manager()
        return cls._instance

    def init_manager(self):
        self.current_lang = 'ko'  # 기본값: 한국어
        self.translations = {
            'ko': {
                'window_title': "RL Maze Solver - Enhanced (RL 미로 탐색기)",
                'dashboard_title': "RL 대시보드 — ",
                'status_label': "상태: ",
                'status_idle': "대기 중 (Idle)",
                'status_training': "학습 진행 중 (Training...)",
                'status_fast_training': "고속 학습 중 (Fast Training...)",
                'status_testing': "최적 경로 테스트 중 (Testing...)",
                'status_editing': "✏️ 편집 모드 (클릭으로 타일 배치)",
                'unknown': "알 수 없음",
                
                'select_algorithm': "알고리즘 선택:",
                'hyperparameters': "하이퍼파라미터:",
                
                'algo_q_learning': "Q-Learning",
                'algo_sarsa': "SARSA",
                'algo_double_q': "Double-Q",
                
                'trap_label_on': "함정: ON",
                'trap_label_off': "함정: OFF",
                
                'slider_alpha': "학습률 (Alpha)",
                'slider_gamma': "감가율 (Gamma)",
                'slider_fps': "학습 속도 (FPS)",
                'slider_size': "미로 크기 (Size)",
                
                'btn_start': "▶ 학습 시작",
                'btn_pause': "⏸ 일시 정지",
                'btn_fast': "⚡ 고속학습 100",
                'btn_test': "🧪 테스트",
                'btn_reset': "🔄 에이전트 및 Q-Table 초기화",
                
                'btn_edit': "🖊 편집 모드",
                'btn_random': "🎲 랜덤 미로",
                'btn_save': "💾 저장",
                'btn_load': "📂 불러오기",
                
                'brush_wall': "벽",
                'brush_path': "길",
                'brush_trap': "함정",
                'brush_bonus': "보너스",
                'brush_start': "시작",
                'brush_goal': "목적지",
                'brush_info': "브러시: {name} | 우클릭=지우기",
                
                'tip_q': "[Q] Q-값 표시 전환",
                'graph_title': "학습 곡선 (에피소드별 Steps)",
                'graph_no_data': "데이터 부족...",
                
                'metric_episode': "에피소드",
                'metric_step': "스텝",
                'metric_success': "성공",
                'metric_best': "최단",
                'metric_epsilon': "ε",
                
                # 설정창
                'settings_title': "설정 (Settings)",
                'settings_lang': "언어 (Language)",
                'settings_theme': "테마 (Theme)",
                'settings_sound': "사운드 효과 (Sound)",
                'settings_grid': "격자선 표시 (Grid)",
                'settings_particles': "파티클 효과 (Particles)",
                'settings_q_overlay': "Q-값 표시 (Q-Overlay)",
                'settings_close': "닫기 (Close)",
                
                'theme_dark': "다크 (Dark)",
                'theme_classic': "클래식 (Classic)",
                'theme_light': "라이트 (Light)",
                
                'state_on': "켜짐",
                'state_off': "꺼짐"
            },
            'en': {
                'window_title': "RL Maze Solver - Enhanced",
                'dashboard_title': "RL Dashboard — ",
                'status_label': "Status: ",
                'status_idle': "Idle",
                'status_training': "Training...",
                'status_fast_training': "Fast Training...",
                'status_testing': "Testing...",
                'status_editing': "✏️ Edit Mode (Click to place tile)",
                'unknown': "Unknown",
                
                'select_algorithm': "Select Algorithm:",
                'hyperparameters': "Hyperparameters:",
                
                'algo_q_learning': "Q-Learning",
                'algo_sarsa': "SARSA",
                'algo_double_q': "Double-Q",
                
                'trap_label_on': "Trap: ON",
                'trap_label_off': "Trap: OFF",
                
                'slider_alpha': "Learning Rate (Alpha)",
                'slider_gamma': "Discount Factor (Gamma)",
                'slider_fps': "Training Speed (FPS)",
                'slider_size': "Maze Size (Size)",
                
                'btn_start': "▶ Start Training",
                'btn_pause': "⏸ Pause",
                'btn_fast': "⚡ Fast Train 100",
                'btn_test': "🧪 Test Agent",
                'btn_reset': "🔄 Reset Agent & Q-Table",
                
                'btn_edit': "🖊 Edit Mode",
                'btn_random': "🎲 Random Maze",
                'btn_save': "💾 Save",
                'btn_load': "📂 Load",
                
                'brush_wall': "Wall",
                'brush_path': "Path",
                'brush_trap': "Trap",
                'brush_bonus': "Bonus",
                'brush_start': "Start",
                'brush_goal': "Goal",
                'brush_info': "Brush: {name} | Right Click=Erase",
                
                'tip_q': "[Q] Toggle Q-Value Display",
                'graph_title': "Learning Curve (Steps per Episode)",
                'graph_no_data': "Not enough data...",
                
                'metric_episode': "Episode",
                'metric_step': "Step",
                'metric_success': "Success",
                'metric_best': "Best",
                'metric_epsilon': "ε",
                
                # Settings modal
                'settings_title': "Settings",
                'settings_lang': "Language",
                'settings_theme': "Theme",
                'settings_sound': "Sound Effects",
                'settings_grid': "Show Grid Lines",
                'settings_particles': "Particles",
                'settings_q_overlay': "Q-Overlay",
                'settings_close': "Close",
                
                'theme_dark': "Dark",
                'theme_classic': "Classic",
                'theme_light': "Light",
                
                'state_on': "ON",
                'state_off': "OFF"
            }
        }

    def set_language(self, lang):
        if lang in self.translations:
            self.current_lang = lang

    def get(self, key, default=None, **kwargs):
        lang_dict = self.translations.get(self.current_lang, self.translations['ko'])
        val = lang_dict.get(key, default if default is not None else key)
        if kwargs:
            return val.format(**kwargs)
        return val

# 전역 인스턴스 제공
i18n = I18NManager()

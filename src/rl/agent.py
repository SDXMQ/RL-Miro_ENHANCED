import numpy as np
import random

class RLAgent:
    """
    강화학습 에이전트 클래스
    지원 알고리즘: q_learning, sarsa, double_q
    """
    def __init__(self, state_space_shape, action_space_size=4,
                 alpha=0.15, gamma=0.85, epsilon=0.999,
                 min_epsilon=0.05, decay_rate=0.9995,
                 algorithm='q_learning'):
        self.height, self.width = state_space_shape
        self.action_space_size = action_space_size
        self.algorithm = algorithm

        # Q-테이블 초기화
        if algorithm == 'double_q':
            # Double Q-러닝: 두 개의 독립적인 Q-테이블 사용
            self.q_table_a = np.zeros((self.height, self.width, self.action_space_size))
            self.q_table_b = np.zeros((self.height, self.width, self.action_space_size))
        else:
            # Q-러닝, SARSA: 단일 Q-테이블 (세로, 가로, 행동개수)
            self._q_table = np.zeros((self.height, self.width, self.action_space_size))

        # 하이퍼파라미터 설정
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    @property
    def q_table(self):
        """Double Q-러닝일 경우 두 테이블의 합을 반환 (행동 선택 및 시각화용)"""
        if self.algorithm == 'double_q':
            return self.q_table_a + self.q_table_b
        return self._q_table

    @q_table.setter
    def q_table(self, value):
        """단일 Q-테이블 모드에서의 setter"""
        if self.algorithm != 'double_q':
            self._q_table = value

    def get_action(self, state, valid_actions):
        """
        epsilon-greedy 탐색 전략에 기반하여 행동을 선택합니다.
        """
        if not valid_actions:
            return random.randint(0, self.action_space_size - 1)

        x, y = state

        # 1. 탐험 (Exploration)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # 2. 활용 (Exploitation) - Q값이 가장 높은 유효한 행동 선택
        combined_q = self.q_table
        q_values = [combined_q[x, y, a] for a in valid_actions]
        max_q = max(q_values)

        # 최대 Q값을 가진 행동이 여러 개일 경우 그 중 무작위 선택 (쏠림 방지)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state, next_valid_actions, next_action=None):
        """
        선택된 알고리즘에 따라 Q-테이블을 업데이트합니다.
        - q_learning: Q(s,a) += α * (r + γ * max Q(s',a') - Q(s,a))
        - sarsa:      Q(s,a) += α * (r + γ * Q(s',a') - Q(s,a)), a'는 next_action
        - double_q:   무작위로 테이블 A/B 중 하나를 선택하여 교차 업데이트
        """
        x, y = state
        nx, ny = next_state

        if self.algorithm == 'q_learning':
            self._update_q_learning(x, y, action, reward, nx, ny, next_valid_actions)
        elif self.algorithm == 'sarsa':
            self._update_sarsa(x, y, action, reward, nx, ny, next_action)
        elif self.algorithm == 'double_q':
            self._update_double_q(x, y, action, reward, nx, ny, next_valid_actions)

    def _update_q_learning(self, x, y, action, reward, nx, ny, next_valid_actions):
        """표준 Q-러닝 업데이트"""
        # 다음 상태의 최대 Q값 계산 (유효한 행동 중에서만)
        if next_valid_actions:
            max_next_q = max([self._q_table[nx, ny, a] for a in next_valid_actions])
        else:
            max_next_q = 0.0

        # Q-테이블 업데이트
        self._q_table[x, y, action] += self.alpha * (
            reward + self.gamma * max_next_q - self._q_table[x, y, action]
        )

    def _update_sarsa(self, x, y, action, reward, nx, ny, next_action):
        """SARSA 업데이트: 실제 선택된 다음 행동의 Q값 사용"""
        # 다음 행동의 Q값 (next_action이 None이면 종료 상태로 간주)
        if next_action is not None:
            next_q = self._q_table[nx, ny, next_action]
        else:
            next_q = 0.0

        # Q-테이블 업데이트
        self._q_table[x, y, action] += self.alpha * (
            reward + self.gamma * next_q - self._q_table[x, y, action]
        )

    def _update_double_q(self, x, y, action, reward, nx, ny, next_valid_actions):
        """Double Q-러닝 업데이트: 두 테이블 교차 사용으로 과대추정 편향 방지"""
        if not next_valid_actions:
            next_q = 0.0
            # 어느 테이블이든 동일하게 업데이트
            if random.random() < 0.5:
                self.q_table_a[x, y, action] += self.alpha * (
                    reward + self.gamma * next_q - self.q_table_a[x, y, action]
                )
            else:
                self.q_table_b[x, y, action] += self.alpha * (
                    reward + self.gamma * next_q - self.q_table_b[x, y, action]
                )
            return

        if random.random() < 0.5:
            # 테이블 A 업데이트: A로 최적 행동 선택, B의 값으로 타겟 계산
            best_action = max(next_valid_actions, key=lambda a: self.q_table_a[nx, ny, a])
            next_q = self.q_table_b[nx, ny, best_action]
            self.q_table_a[x, y, action] += self.alpha * (
                reward + self.gamma * next_q - self.q_table_a[x, y, action]
            )
        else:
            # 테이블 B 업데이트: B로 최적 행동 선택, A의 값으로 타겟 계산
            best_action = max(next_valid_actions, key=lambda a: self.q_table_b[nx, ny, a])
            next_q = self.q_table_a[nx, ny, best_action]
            self.q_table_b[x, y, action] += self.alpha * (
                reward + self.gamma * next_q - self.q_table_b[x, y, action]
            )

    def decay_epsilon(self):
        """
        에피소드 종료 후 탐험률(epsilon)을 점진적으로 감쇄시킵니다.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def reset(self):
        """
        모든 Q-테이블을 0으로 재초기화합니다 (형상 유지).
        """
        if self.algorithm == 'double_q':
            self.q_table_a = np.zeros((self.height, self.width, self.action_space_size))
            self.q_table_b = np.zeros((self.height, self.width, self.action_space_size))
        else:
            self._q_table = np.zeros((self.height, self.width, self.action_space_size))

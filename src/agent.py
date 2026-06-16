import numpy as np
import random

class QLearningAgent:
    """
    Q-러닝 강화학습 에이전트 클래스
    """
    def __init__(self, state_space_shape, action_space_size=4, 
                 alpha=0.15, gamma=0.85, epsilon=0.999, 
                 min_epsilon=0.05, decay_rate=0.9995):
        self.height, self.width = state_space_shape
        self.action_space_size = action_space_size
        
        # Q-테이블 (세로, 가로, 행동개수) 크기의 0행렬로 초기화
        self.q_table = np.zeros((self.height, self.width, self.action_space_size))
        
        # 하이퍼파라미터 설정
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

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
        q_values = [self.q_table[x, y, a] for a in valid_actions]
        max_q = max(q_values)
        
        # 최대 Q값을 가진 행동이 여러 개일 경우 그 중 무작위 선택 (쏠림 방지)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state, next_valid_actions):
        """
        Q-러닝 공식에 따라 Q-테이블을 업데이트합니다.
        Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
        """
        x, y = state
        nx, ny = next_state
        
        # 다음 상태의 최대 Q값 계산 (유효한 행동 중에서만)
        if next_valid_actions:
            max_next_q = max([self.q_table[nx, ny, a] for a in next_valid_actions])
        else:
            max_next_q = 0.0

        # Q-테이블 업데이트
        self.q_table[x, y, action] += self.alpha * (
            reward + self.gamma * max_next_q - self.q_table[x, y, action]
        )

    def decay_epsilon(self):
        """
        에피소드 종료 후 탐험률(epsilon)을 점진적으로 감쇄시킵니다.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

import random
import json
import ast
from typing import Tuple, Dict, Any, Optional

class SarsaAgent:
    """
    Thuật toán SARSA Agent cho Pacman Mini.
    Đây là thuật toán On-Policy: cập nhật dựa trên hành động thực tế tiếp theo (a').
    Nhiệm vụ của TV3 là bảo trì thuật toán này.
    """

    def __init__(
        self,
        action_space_size: int = 4,
        alpha: float = 0.1,      # Tốc độ học (Learning rate)
        gamma: float = 0.95,     # Hệ số chiết khấu (Discount factor)
        epsilon: float = 1.0,    # Khám phá ban đầu (Exploration rate)
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995
    ):
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def get_q_values(self, state: Tuple[int, int, int, int]) -> list:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_space_size
        return self.q_table[state]

    def choose_action(self, state: Tuple[int, int, int, int], evaluation: bool = False) -> int:
        """
        Chọn hành động theo Epsilon-Greedy.
        """
        if evaluation or random.random() > self.epsilon:
            q_values = self.get_q_values(state)
            max_q = max(q_values)
            actions_with_max_q = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(actions_with_max_q)
        else:
            return random.randint(0, self.action_space_size - 1)

    def update(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
        next_action: int,      # Khác biệt của SARSA: Cần truyền vào hành động tiếp theo
        done: bool
    ) -> float:
        """
        Cập nhật Q-table theo công thức SARSA (On-policy):
        Q(s, a) = Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
        """
        q_values = self.get_q_values(state)
        current_q = q_values[action]
        
        if done:
            td_target = reward
        else:
            # Lấy Q-value của hành động tiếp theo thực tế sẽ chọn (next_action)
            next_q_values = self.get_q_values(next_state)
            td_target = reward + self.gamma * next_q_values[next_action]
            
        td_error = td_target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error
        
        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_policy(self, filepath: str) -> None:
        serialized_q_table = {str(k): v for k, v in self.q_table.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serialized_q_table, f, indent=4)

    def load_policy(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as f:
            serialized_q_table = json.load(f)
        self.q_table = {ast.literal_eval(k): v for k, v in serialized_q_table.items()}

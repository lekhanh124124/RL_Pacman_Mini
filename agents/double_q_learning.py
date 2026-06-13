import random
import json
import ast
from typing import Tuple, Dict, Any, Optional

class DoubleQLearningAgent:
    """
    Thuật toán Double Q-Learning Agent cho Pacman Mini.
    Giải quyết lệch lạc lạc quan thái quá (Overestimation Bias) bằng cách sử dụng 2 bảng Q độc lập.
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
        
        # Sử dụng hai bảng Q-table độc lập
        self.q_table_a = {}
        self.q_table_b = {}

    def get_q_values(self, state: Tuple[int, int, int, int], table: str = 'a') -> list:
        """
        Lấy danh sách các giá trị Q tại trạng thái từ bảng tương ứng ('a' hoặc 'b').
        """
        q_table = self.q_table_a if table == 'a' else self.q_table_b
        if state not in q_table:
            q_table[state] = [0.0] * self.action_space_size
        return q_table[state]

    def choose_action(self, state: Tuple[int, int, int, int], evaluation: bool = False) -> int:
        """
        Chọn hành động theo chiến lược Epsilon-Greedy dựa trên tổng của hai bảng Q.
        """
        if evaluation or random.random() > self.epsilon:
            # Ước lượng tốt nhất bằng cách cộng hai bảng Q
            q_a = self.get_q_values(state, 'a')
            q_b = self.get_q_values(state, 'b')
            q_values = [a + b for a, b in zip(q_a, q_b)]
            
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
        done: bool
    ) -> float:
        """
        Cập nhật một trong hai bảng Q ngẫu nhiên với xác suất 50/50.
        """
        if random.random() < 0.5:
            # Cập nhật Q_A sử dụng Q_B để đánh giá
            q_a = self.get_q_values(state, 'a')
            current_q = q_a[action]
            
            if done:
                td_target = reward
            else:
                # Chọn a* = argmax_a Q_A(s', a)
                next_q_a = self.get_q_values(next_state, 'a')
                max_action = next_q_a.index(max(next_q_a))
                # Đánh giá Q_B(s', a*)
                next_q_b = self.get_q_values(next_state, 'b')
                td_target = reward + self.gamma * next_q_b[max_action]
                
            td_error = td_target - current_q
            self.q_table_a[state][action] = current_q + self.alpha * td_error
        else:
            # Cập nhật Q_B sử dụng Q_A để đánh giá
            q_b = self.get_q_values(state, 'b')
            current_q = q_b[action]
            
            if done:
                td_target = reward
            else:
                # Chọn b* = argmax_b Q_B(s', b)
                next_q_b = self.get_q_values(next_state, 'b')
                max_action = next_q_b.index(max(next_q_b))
                # Đánh giá Q_A(s', b*)
                next_q_a = self.get_q_values(next_state, 'a')
                td_target = reward + self.gamma * next_q_a[max_action]
                
            td_error = td_target - current_q
            self.q_table_b[state][action] = current_q + self.alpha * td_error
            
        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_policy(self, filepath: str) -> None:
        serialized_q_table_a = {str(k): v for k, v in self.q_table_a.items()}
        serialized_q_table_b = {str(k): v for k, v in self.q_table_b.items()}
        
        data = {
            'q_table_a': serialized_q_table_a,
            'q_table_b': serialized_q_table_b
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def load_policy(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.q_table_a = {ast.literal_eval(k): v for k, v in data['q_table_a'].items()}
        self.q_table_b = {ast.literal_eval(k): v for k, v in data['q_table_b'].items()}

import random
import json
import ast
from typing import Tuple, Dict, Any, Optional

class QLearningAgent:
    """
    Thuật toán Q-Learning Agent cho Pacman Mini.
    Nhiệm vụ của TV3 là hoàn thiện và bảo trì thuật toán này.
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
        
        # Bảng Q-table biểu diễn dưới dạng Dictionary: {state_tuple: [q_values_for_actions]}
        # Ví dụ: {(1, 4, 2, 0): [0.0, 0.0, 0.0, 0.0]}
        self.q_table = {}

    def get_q_values(self, state: Tuple[int, int, int, int]) -> list:
        """
        Lấy danh sách các giá trị Q tại trạng thái hiện tại.
        Nếu trạng thái chưa từng gặp, tự động khởi tạo bằng 0.
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_space_size
        return self.q_table[state]

    def choose_action(self, state: Tuple[int, int, int, int], evaluation: bool = False) -> int:
        """
        Chọn hành động theo chiến lược Epsilon-Greedy.
        
        Args:
            state: Trạng thái nén hiện tại.
            evaluation: Nếu True thì epsilon = 0 (chế độ đánh giá, chỉ lấy argmax).
        """
        # Chế độ đánh giá (Evaluation) hoặc ngẫu nhiên lớn hơn Epsilon
        if evaluation or random.random() > self.epsilon:
            q_values = self.get_q_values(state)
            # Chọn hành động có giá trị Q cao nhất (argmax ngẫu nhiên nếu bằng nhau)
            max_q = max(q_values)
            actions_with_max_q = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(actions_with_max_q)
        else:
            # Chọn hành động ngẫu nhiên (Exploration)
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
        Cập nhật Q-table theo phương trình Bellman:
        Q(s, a) = Q(s, a) + alpha * [r + gamma * max_a(Q(s', a)) - Q(s, a)]
        
        Returns:
            td_error (độ lỗi Temporal Difference) để ghi log.
        """
        q_values = self.get_q_values(state)
        current_q = q_values[action]
        
        if done:
            # Nếu s_t+1 là trạng thái kết thúc (Terminal), target chỉ là reward r
            td_target = reward
        else:
            # TD Target = r + gamma * max_a(Q(s', a))
            next_q_values = self.get_q_values(next_state)
            td_target = reward + self.gamma * max(next_q_values)
            
        td_error = td_target - current_q
        
        # Cập nhật Q-value
        self.q_table[state][action] = current_q + self.alpha * td_error
        
        return td_error

    def decay_epsilon(self) -> None:
        """
        Giảm dần độ khám phá sau mỗi episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_policy(self, filepath: str) -> None:
        """
        Lưu bảng Q-table đã học ra file JSON để TV5 nhúng vào web demo chạy live.
        """
        # Chuyển đổi key từ tuple sang string để ghi được file JSON
        serialized_q_table = {str(k): v for k, v in self.q_table.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serialized_q_table, f, indent=4)

    def load_policy(self, filepath: str) -> None:
        """
        Tải Q-table đã lưu từ file JSON.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            serialized_q_table = json.load(f)
        
        # Chuyển đổi key từ string trở lại thành tuple
        self.q_table = {ast.literal_eval(k): v for k, v in serialized_q_table.items()}

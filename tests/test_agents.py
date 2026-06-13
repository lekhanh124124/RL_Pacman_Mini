import unittest
import sys
import os

# Thêm thư mục src vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.double_q_learning import DoubleQLearningAgent

class TestAgents(unittest.TestCase):
    """
    Unit Tests cho toàn bộ hệ thống Agent & Thuật toán.
    Đảm bảo tất cả tác nhân hoạt động đúng theo giao thức API.
    """

    def setUp(self):
        self.state = (12, 4, 1, 2)  # (agent_pos, ghost_zone, nearest_coin_dir, coin_count_bin)
        self.next_state = (13, 4, 1, 2)

    def test_random_agent(self):
        """Kiểm tra tác nhân ngẫu nhiên."""
        agent = RandomAgent()
        for _ in range(20):
            action = agent.choose_action(self.state)
            self.assertIn(action, [0, 1, 2, 3])

    def test_heuristic_agent_evasion(self):
        """Kiểm tra quy tắc né tránh của Heuristic Agent khi Ghost ở kề bên."""
        agent = HeuristicAgent()
        
        # Ghost kề trên (0) -> Phải đi xuống (2)
        state = (12, 0, 1, 2)
        self.assertEqual(agent.choose_action(state), 2)
        
        # Ghost kề phải (1) -> Phải đi trái (3)
        state = (12, 1, 1, 2)
        self.assertEqual(agent.choose_action(state), 3)

    def test_heuristic_agent_greedy(self):
        """Kiểm tra quy tắc tìm xu của Heuristic Agent khi an toàn."""
        agent = HeuristicAgent()
        
        # An toàn (4) và xu kề phải (1) -> Đi phải (1)
        state = (12, 4, 1, 2)
        self.assertEqual(agent.choose_action(state), 1)

    def test_q_learning_agent_update(self):
        """Kiểm tra cập nhật Bellman của Q-Learning."""
        agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.0)
        
        # Cập nhật thử trạng thái thường
        td_error = agent.update(self.state, 1, 5.0, self.next_state, done=False)
        q_vals = agent.get_q_values(self.state)
        
        # Q(s, a) = 0 + 0.1 * (5.0 + 0.9 * 0 - 0) = 0.5
        self.assertAlmostEqual(q_vals[1], 0.5)
        self.assertAlmostEqual(td_error, 5.0)

        # Cập nhật thử trạng thái kết thúc (Terminal)
        td_error_term = agent.update(self.state, 2, -50.0, self.next_state, done=True)
        q_vals_term = agent.get_q_values(self.state)
        # Q(s, a) = 0 + 0.1 * (-50.0 - 0) = -5.0
        self.assertAlmostEqual(q_vals_term[2], -5.0)

    def test_sarsa_agent_update(self):
        """Kiểm tra cập nhật On-policy của SARSA."""
        agent = SarsaAgent(alpha=0.1, gamma=0.9, epsilon=0.0)
        
        # Cập nhật thử trạng thái thường
        td_error = agent.update(self.state, 1, 5.0, self.next_state, next_action=0, done=False)
        q_vals = agent.get_q_values(self.state)
        # Q(s, a) = 0 + 0.1 * (5.0 + 0.9 * 0 - 0) = 0.5
        self.assertAlmostEqual(q_vals[1], 0.5)
        
        # Cập nhật thử trạng thái kết thúc
        td_error_term = agent.update(self.state, 2, 30.0, self.next_state, next_action=0, done=True)
        q_vals_term = agent.get_q_values(self.state)
        # Q(s, a) = 0 + 0.1 * (30.0 - 0) = 3.0
        self.assertAlmostEqual(q_vals_term[2], 3.0)

    def test_double_q_learning_agent_update(self):
        """Kiểm tra cập nhật song song của Double Q-Learning."""
        agent = DoubleQLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.0)
        
        # Cập nhật thử và đảm bảo hoạt động không bị lỗi cú pháp
        td_error = agent.update(self.state, 1, 5.0, self.next_state, done=False)
        self.assertTrue(isinstance(td_error, float))

        # Kiểm tra lưu và tải chính sách hoạt động đúng đắn
        test_file = "test_double_q_policy.json"
        try:
            agent.save_policy(test_file)
            new_agent = DoubleQLearningAgent()
            new_agent.load_policy(test_file)
            self.assertEqual(new_agent.q_table_a.keys(), agent.q_table_a.keys())
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

if __name__ == '__main__':
    unittest.main()

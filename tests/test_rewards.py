import unittest
import sys
import os

# Thêm thư mục src vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import PacmanMiniEnv, ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT

class TestRewardShaping(unittest.TestCase):
    """
    Unit Tests cho Reward Shaping.
    """

    def setUp(self):
        self.env = PacmanMiniEnv(width=7, height=7)

    def test_default_step_penalty(self):
        """Kiểm tra phạt bước đi thông thường (-1.0)."""
        self.env.reset()
        self.env.pacman_pos = (1, 1)
        self.env.ghost_pos = (5, 5)
        self.env.coins = {(1, 4)} # coins ở xa
        
        _, reward, terminated, _, _ = self.env.step(ACTION_RIGHT) # Di chuyển sang (1, 2)
        self.assertEqual(reward, -1.0)
        self.assertFalse(terminated)

    def test_wall_penalty(self):
        """Kiểm tra phạt đụng tường (-5.0) và không di chuyển."""
        self.env.reset()
        self.env.pacman_pos = (1, 1)
        self.env.ghost_pos = (5, 5)
        
        _, reward, terminated, _, info = self.env.step(ACTION_UP) # Đâm lên tường (0, 1)
        self.assertEqual(reward, -5.0)
        self.assertEqual(self.env.pacman_pos, (1, 1))
        self.assertTrue(info['hit_wall'])
        self.assertFalse(terminated)

    def test_coin_eat_reward(self):
        """Kiểm tra thưởng ăn xu (+5.0) khi vẫn còn xu khác trên bản đồ."""
        self.env.reset()
        self.env.pacman_pos = (1, 1)
        self.env.ghost_pos = (5, 5)
        self.env.coins = {(1, 2), (1, 4)} # Có 2 đồng xu
        
        _, reward, terminated, _, _ = self.env.step(ACTION_RIGHT) # Di chuyển sang (1, 2) ăn xu
        self.assertEqual(reward, 5.0)
        self.assertNotIn((1, 2), self.env.coins)
        self.assertEqual(len(self.env.coins), 1)
        self.assertFalse(terminated)

    def test_win_reward(self):
        """Kiểm tra thưởng chiến thắng (+30.0) và kết thúc game khi ăn đồng xu cuối cùng."""
        self.env.reset()
        self.env.pacman_pos = (1, 1)
        self.env.ghost_pos = (5, 5)
        self.env.coins = {(1, 2)} # Chỉ có 1 đồng xu duy nhất
        
        _, reward, terminated, _, _ = self.env.step(ACTION_RIGHT) # Di chuyển sang (1, 2) ăn sạch xu
        self.assertEqual(reward, 30.0)
        self.assertEqual(len(self.env.coins), 0)
        self.assertTrue(terminated)

    def test_caught_by_ghost_pacman_move(self):
        """Kiểm tra phạt bị ma bắt (-50.0) khi Pacman chủ động di chuyển đâm vào Ghost."""
        self.env.reset()
        self.env.pacman_pos = (1, 1)
        self.env.ghost_pos = (1, 2) # Ghost ở ngay bên phải
        
        _, reward, terminated, _, info = self.env.step(ACTION_RIGHT) # Pacman đâm vào Ghost
        self.assertEqual(reward, -50.0)
        self.assertTrue(terminated)
        self.assertTrue(info['caught'])

    def test_caught_by_ghost_ghost_move(self):
        """Kiểm tra phạt bị ma bắt (-50.0) khi Ghost di chuyển đâm vào Pacman."""
        self.env.reset()
        self.env.pacman_pos = (1, 1)
        self.env.ghost_pos = (1, 2)
        # Để đảm bảo Ghost chắc chắn đi vào (1, 1), chúng ta mock hàm move_ghost của env
        def mock_move_ghost():
            self.env.ghost_pos = (1, 1)
        self.env.move_ghost = mock_move_ghost
        
        # Pacman chọn hành động đứng yên hoặc đi đụng tường
        _, reward, terminated, _, info = self.env.step(ACTION_UP)
        self.assertEqual(reward, -50.0)
        self.assertTrue(terminated)
        self.assertTrue(info['caught'])
        
    def test_custom_rewards_shaping(self):
        """Kiểm tra xem các phần thưởng tùy chỉnh (custom rewards) có được áp dụng chính xác không."""
        custom_rewards = {
            'coin': 10.0,
            'win': 100.0,
            'caught': -200.0,
            'wall': -2.0,
            'step': -0.5
        }
        env = PacmanMiniEnv(width=7, height=7, rewards=custom_rewards)
        
        # Test step penalty
        env.reset()
        env.pacman_pos = (1, 1)
        env.ghost_pos = (5, 5)
        env.coins = {(1, 4)}
        _, reward, terminated, _, _ = env.step(ACTION_RIGHT)
        self.assertEqual(reward, -0.5)
        
        # Test wall penalty
        env.pacman_pos = (1, 1)
        _, reward, _, _, _ = env.step(ACTION_UP)
        self.assertEqual(reward, -2.0)
        
        # Test coin reward
        env.pacman_pos = (1, 1)
        env.coins = {(1, 2), (1, 4)}
        _, reward, _, _, _ = env.step(ACTION_RIGHT)
        self.assertEqual(reward, 10.0)
        
        # Test win reward
        env.pacman_pos = (1, 1)
        env.coins = {(1, 2)}
        _, reward, terminated, _, _ = env.step(ACTION_RIGHT)
        self.assertEqual(reward, 100.0)
        self.assertTrue(terminated)
        
        # Test caught penalty
        env.reset()
        env.pacman_pos = (1, 1)
        env.ghost_pos = (1, 2)
        _, reward, terminated, _, _ = env.step(ACTION_RIGHT)
        self.assertEqual(reward, -200.0)
        self.assertTrue(terminated)

if __name__ == '__main__':
    unittest.main()

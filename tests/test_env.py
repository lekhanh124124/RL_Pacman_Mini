import unittest
import sys
import os

# Thêm thư mục src vào PATH để import được các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import PacmanMiniEnv, ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT

class TestPacmanMiniEnv(unittest.TestCase):
    """
    Unit Tests cho môi trường Pacman Mini 7x7.
    """

    def setUp(self):
        self.env = PacmanMiniEnv(width=7, height=7, seed=42)

    def test_initial_state(self):
        """Kiểm tra xem trạng thái khởi tạo có đúng 7x7 và đúng số xu không."""
        state = self.env.reset(seed=42)
        self.assertEqual(len(state), 4)
        self.assertTrue(all(isinstance(x, int) for x in state))
        
        self.assertEqual(self.env.pacman_pos, (1, 1))
        self.assertEqual(self.env.ghost_pos, (7, 7))
        self.assertEqual(len(self.env.coins), 5)
        self.assertEqual(self.env.current_step, 0)

    def test_step_return_format(self):
        """Kiểm tra định dạng dữ liệu trả về từ hàm step()."""
        state = self.env.reset()
        next_state, reward, terminated, truncated, info = self.env.step(ACTION_RIGHT)
        
        self.assertEqual(len(next_state), 4)
        self.assertTrue(isinstance(reward, (int, float)))
        self.assertTrue(isinstance(terminated, bool))
        self.assertTrue(isinstance(truncated, bool))
        self.assertTrue(isinstance(info, dict))
        self.assertIn('pacman_pos', info)
        self.assertIn('ghost_pos', info)
        self.assertIn('coins_left', info)
        self.assertIn('hit_wall', info)
        self.assertIn('caught', info)

    def test_wall_collision(self):
        """Kiểm tra va chạm tường: Pacman đứng yên và nhận phạt đụng tường (-5)."""
        self.env.reset()
        # Pacman ở (1,1). Cố gắng đi UP (0) sẽ đâm vào tường (0,1).
        next_state, reward, terminated, truncated, info = self.env.step(ACTION_UP)
        self.assertEqual(self.env.pacman_pos, (1, 1))
        self.assertEqual(reward, -5.0)
        self.assertTrue(info['hit_wall'])

    def test_normal_step(self):
        """Kiểm tra di chuyển bình thường không đụng tường không ăn xu."""
        self.env.reset()
        # Pacman ở (1, 1). Đi sang phải -> (1, 2) (không có xu, không có ma)
        next_state, reward, terminated, truncated, info = self.env.step(ACTION_RIGHT)
        self.assertEqual(self.env.pacman_pos, (1, 2))
        # Không có xu ở (1, 2) nên nhận phạt bước đi -1.0
        self.assertEqual(reward, -1.0)
        self.assertFalse(info['hit_wall'])

    def test_deterministic_seed(self):
        """Kiểm tra xem thiết lập seed có tạo ra kết quả hoàn toàn giống nhau không."""
        env1 = PacmanMiniEnv(width=7, height=7, seed=123)
        env2 = PacmanMiniEnv(width=7, height=7, seed=123)
        
        state1 = env1.reset(seed=123)
        state2 = env2.reset(seed=123)
        self.assertEqual(state1, state2)
        
        # Chạy 5 bước giống nhau, kiểm tra xem vị trí của Pacman và Ghost có trùng khớp hoàn toàn không
        for _ in range(5):
            res1 = env1.step(ACTION_RIGHT)
            res2 = env2.step(ACTION_RIGHT)
            self.assertEqual(res1[0], res2[0]) # state nén
            self.assertEqual(res1[1], res2[1]) # reward
            self.assertEqual(res1[2], res2[2]) # terminated
            self.assertEqual(res1[3], res2[3]) # truncated
            self.assertEqual(res1[4]['ghost_pos'], res2[4]['ghost_pos']) # vị trí ma

    def test_5x5_initial_state(self):
        """Kiểm tra xem map 5x5 walkable (ma trận 7x7 thực tế) khởi tạo đúng vị trí mặc định không."""
        env = PacmanMiniEnv(width=5, height=5, seed=42)
        self.assertEqual(env.width, 7)
        self.assertEqual(env.height, 7)
        self.assertEqual(env.pacman_pos, (1, 1))
        self.assertEqual(env.ghost_pos, (5, 5))
        self.assertEqual(len(env.coins), 5)

    def test_no_overlap_when_mixed_randomization(self):
        """Kiểm tra xem khi trộn cấu hình (ngẫu nhiên hóa Pacman nhưng giữ tĩnh Ghost) thì Pacman và Ghost có bị trùng không."""
        for seed in range(100):
            env = PacmanMiniEnv(width=6, height=6, seed=seed, randomize_pacman=True, randomize_ghost=False)
            self.assertNotEqual(env.pacman_pos, env.ghost_pos)
            self.assertNotIn(env.pacman_pos, env.coins)
            self.assertNotIn(env.ghost_pos, env.coins)

if __name__ == '__main__':
    unittest.main()

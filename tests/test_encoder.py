import unittest
import sys
import os

# Thêm thư mục src vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import PacmanMiniEnv, GHOST_ZONE_MAP, COIN_DIR_MAP, COIN_COUNT_MAP

class TestStateEncoder(unittest.TestCase):
    """
    Unit Tests cho State Encoder.
    """

    def setUp(self):
        self.env = PacmanMiniEnv(width=7, height=7)

    def test_state_encoder_range(self):
        """Kiểm tra xem các chỉ số nén có nằm đúng trong phạm vi ánh xạ định trước không."""
        state = self.env.reset()
        agent_pos, ghost_zone, coin_dir, coin_count = state
        
        # 1. Kiểm tra vị trí agent nằm trong lưới [0, 48]
        self.assertTrue(0 <= agent_pos < (self.env.width * self.env.height))
        
        # 2. Kiểm tra ghost zone nằm trong GHOST_ZONE_MAP
        self.assertIn(ghost_zone, GHOST_ZONE_MAP.values())
        
        # 3. Kiểm tra nearest coin dir nằm trong COIN_DIR_MAP
        self.assertIn(coin_dir, COIN_DIR_MAP.values())
        
        # 4. Kiểm tra coin count bin nằm trong COIN_COUNT_MAP
        self.assertIn(coin_count, COIN_COUNT_MAP.values())

    def test_ghost_zone_directions(self):
        """Kiểm tra xem ghost_zone được mã hóa chuẩn xác khi Ghost ở kề Pacman ở 4 hướng."""
        self.env.reset()
        self.env.pacman_pos = (3, 3)
        
        # Ma kề trên
        self.env.ghost_pos = (2, 3)
        _, ghost_zone, _, _ = self.env.state_encoder()
        self.assertEqual(ghost_zone, GHOST_ZONE_MAP['near_up'])
        
        # Ma kề phải
        self.env.ghost_pos = (3, 4)
        _, ghost_zone, _, _ = self.env.state_encoder()
        self.assertEqual(ghost_zone, GHOST_ZONE_MAP['near_right'])
        
        # Ma kề dưới
        self.env.ghost_pos = (4, 3)
        _, ghost_zone, _, _ = self.env.state_encoder()
        self.assertEqual(ghost_zone, GHOST_ZONE_MAP['near_down'])
        
        # Ma kề trái
        self.env.ghost_pos = (3, 2)
        _, ghost_zone, _, _ = self.env.state_encoder()
        self.assertEqual(ghost_zone, GHOST_ZONE_MAP['near_left'])
        
        # Ma ở xa
        self.env.ghost_pos = (1, 1)
        _, ghost_zone, _, _ = self.env.state_encoder()
        self.assertEqual(ghost_zone, GHOST_ZONE_MAP['far'])

    def test_nearest_coin_direction_bfs(self):
        """Kiểm tra xem hướng đi ngắn nhất (BFS) dẫn tới đồng xu gần nhất có chính xác không."""
        self.env.reset()
        self.env.pacman_pos = (1, 1)
        
        # 1. Đặt duy nhất 1 đồng xu tại (1, 3). Hướng gần nhất phải là 'right'
        self.env.coins = {(1, 3)}
        _, _, coin_dir, _ = self.env.state_encoder()
        self.assertEqual(coin_dir, COIN_DIR_MAP['right'])
        
        # 2. Đặt duy nhất 1 đồng xu tại (3, 1). Đường đi ngắn nhất phải đi qua (2, 1) rồi tới (3, 1). Hướng phải là 'down'
        self.env.coins = {(3, 1)}
        _, _, coin_dir, _ = self.env.state_encoder()
        self.assertEqual(coin_dir, COIN_DIR_MAP['down'])
        
        # 3. Không có đồng xu nào
        self.env.coins = set()
        _, _, coin_dir, _ = self.env.state_encoder()
        self.assertEqual(coin_dir, COIN_DIR_MAP['none'])

    def test_state_decoder_roundtrip_and_labels(self):
        """Kiểm tra state_decoder() trả về đúng nhãn và agent_pos khớp với state_encoder()."""
        self.env.reset(seed=42)
        self.env.pacman_pos = (1, 1)
        self.env.ghost_pos = (5, 5)  # far
        self.env.coins = {(1, 2), (1, 4)}  # 2 coins -> medium, nearest dir -> right

        state = self.env.state_encoder()
        decoded = self.env.state_decoder(state)

        self.assertEqual(decoded['agent_pos'], (1, 1))
        self.assertEqual(decoded['ghost_zone'], 'far')
        self.assertEqual(decoded['nearest_coin_dir'], 'right')
        self.assertEqual(decoded['coin_count_bin'], 'medium')

    def test_state_decoder_ghost_zone_near_up(self):
        """Kiểm tra decoder trả về đúng ghost_zone khi ghost kề trên."""
        self.env.reset(seed=42)
        self.env.pacman_pos = (3, 3)
        self.env.ghost_pos = (2, 3)
        state = self.env.state_encoder()
        decoded = self.env.state_decoder(state)
        self.assertEqual(decoded['ghost_zone'], 'near_up')

if __name__ == '__main__':
    unittest.main()

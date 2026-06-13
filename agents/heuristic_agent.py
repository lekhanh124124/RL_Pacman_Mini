from typing import Tuple, List

class HeuristicAgent:
    """
    Heuristic Agent dựa trên luật đơn giản để ra quyết định:
    1. Né Ghost nếu Ghost quá gần (nằm trong vùng nguy hiểm).
    2. Đi về phía đồng xu gần nhất (dựa trên thông tin hướng xu gần nhất).
    
    Nhiệm vụ của TV4 là tinh chỉnh logic của Agent này.
    """
    def __init__(self, action_space_size: int = 4):
        self.action_space_size = action_space_size

    def choose_action(self, state: Tuple[int, int, int, int], evaluation: bool = False) -> int:
        """
        State format: (agent_pos_index, ghost_zone_index, nearest_coin_dir_index, coin_count_bin_index)
        
        Ánh xạ chỉ số (từ file envs/custom_env.py):
        - Action: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        - Ghost Zone: 0: near_up, 1: near_right, 2: near_down, 3: near_left, 4: far
        - Coin Dir: 0: up, 1: right, 2: down, 3: left, 4: none
        """
        agent_pos, ghost_zone, nearest_coin_dir, coin_count_bin = state

        # 1. Luật né Ghost: Nếu Ghost đang ở rất gần Pacman (near_up, near_right, near_down, near_left)
        # Chúng ta chọn hướng đi ngược lại với hướng của con ma để né tránh.
        if ghost_zone == 0:  # Ghost đang ở UP -> Thử đi DOWN (2) hoặc sang hai bên LEFT (3)/RIGHT (1)
            return 2
        elif ghost_zone == 1:  # Ghost đang ở RIGHT -> Thử đi LEFT (3) hoặc UP (0)/DOWN (2)
            return 3
        elif ghost_zone == 2:  # Ghost đang ở DOWN -> Thử đi UP (0) hoặc LEFT (3)/RIGHT (1)
            return 0
        elif ghost_zone == 3:  # Ghost đang ở LEFT -> Thử đi RIGHT (1) hoặc UP (0)/DOWN (2)
            return 1

        # 2. Luật đi tìm Coin: Nếu không có nguy cơ từ Ghost và có hướng xu gần nhất
        # Đi theo hướng của đồng xu gần nhất
        if nearest_coin_dir in [0, 1, 2, 3]:
            return nearest_coin_dir

        # 3. Mặc định: Nếu không khớp luật nào, chọn đi ngẫu nhiên
        import random
        return random.randint(0, self.action_space_size - 1)

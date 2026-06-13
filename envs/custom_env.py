import random
from typing import Tuple, Dict, Any, Optional
from envs.base_env import BaseEnv

# --- ĐỊNH NGHĨA HẰNG SỐ HỢP ĐỒNG API ---
# Hành động (Actions)
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTIONS = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]

# Trạng thái nén (State Indexes)
GHOST_ZONE_MAP = {
    'near_up': 0,
    'near_right': 1,
    'near_down': 2,
    'near_left': 3,
    'far': 4
}

INV_GHOST_ZONE_MAP = {v: k for k, v in GHOST_ZONE_MAP.items()}

COIN_DIR_MAP = {
    'up': 0,
    'right': 1,
    'down': 2,
    'left': 3,
    'none': 4
}

INV_COIN_DIR_MAP = {v: k for k, v in COIN_DIR_MAP.items()}

COIN_COUNT_MAP = {
    'low': 0,      # Ví dụ: 0-1 xu còn lại
    'medium': 1,   # Ví dụ: 2-3 xu còn lại
    'high': 2      # Ví dụ: 4+ xu còn lại
}

INV_COIN_COUNT_MAP = {v: k for k, v in COIN_COUNT_MAP.items()}

class PacmanMiniEnv(BaseEnv):
    """
    Môi trường Pacman Mini (7x7).
    """

    def __init__(self, width: int = 6, height: int = 6, seed: Optional[int] = None, ghost_chase_prob: float = 0.3, randomize_pacman: bool = False, randomize_ghost: bool = False, randomize_coins: bool = False, randomize_walls: bool = False, max_steps: int = 100, num_coins: int = 5, rewards: Optional[Dict[str, float]] = None):
        self.walkable_width = width
        self.walkable_height = height
        self.width = width + 2
        self.height = height + 2
        self.ghost_chase_prob = ghost_chase_prob
        self.randomize_pacman = randomize_pacman
        self.randomize_ghost = randomize_ghost
        self.randomize_coins = randomize_coins
        self.randomize_walls = randomize_walls
        self.init_max_steps = max_steps
        self.num_coins = num_coins
        self.rewards = rewards or {
            'coin': 5.0,
            'win': 30.0,
            'caught': -50.0,
            'wall': -5.0,
            'step': -1.0
        }
        
        # Thiết lập map mặc định cố định dựa trên kích thước
        self.grid = self._get_default_grid()
        
        # Phải đảm bảo kích thước map thực tế khớp với self.width và self.height
        assert len(self.grid) == self.width
        assert len(self.grid[0]) == self.height

        self.seed(seed)
        self.reset()

    def seed(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[int, int, int, int]:
        """
        Khởi tạo lại trạng thái game.
        """
        if seed is not None:
            self.seed(seed)
            
        self.current_step = 0
        self.max_steps = self.init_max_steps
        
        # 1. Khởi động lưới tường mặc định cố định dựa trên kích thước
        self.grid = self._get_default_grid()
        
        # 2. Xử lý trường hợp có ngẫu nhiên hóa Tường (Randomize Walls) trước tiên
        # Nếu randomize_walls là True, chúng ta tạo một lưới rỗng có viền tường
        # và sẽ đặt tường ngẫu nhiên sau khi đặt Pacman, Ghost, Coins.
        if self.randomize_walls:
            self.grid = [[1] * self.height for _ in range(self.width)]
            for r in range(1, self.width - 1):
                for c in range(1, self.height - 1):
                    self.grid[r][c] = 0
        
        # 3. Đặt Pacman
        if self.randomize_pacman:
            # Chọn ô trống bất kỳ trong lưới (tránh tường mặc định nếu không random walls)
            valid_pacman_pos = []
            for r in range(1, self.width - 1):
                for c in range(1, self.height - 1):
                    if self.grid[r][c] == 0:
                        valid_pacman_pos.append((r, c))
            self.pacman_pos = self.rng.choice(valid_pacman_pos) if valid_pacman_pos else (1, 1)
        else:
            self.pacman_pos = (1, 1)
            
        # 4. Đặt Coins
        if self.randomize_coins:
            valid_coin_pos = []
            for r in range(1, self.width - 1):
                for c in range(1, self.height - 1):
                    if self.grid[r][c] == 0 and (r, c) != self.pacman_pos:
                        valid_coin_pos.append((r, c))
            if len(valid_coin_pos) >= 5:
                self.coins = set(self.rng.sample(valid_coin_pos, 5))
            else:
                self.coins = self._get_default_coins()
        else:
            self.coins = self._get_default_coins()
            
            # Đảm bảo không trùng với Pacman nếu Pacman được đặt ngẫu nhiên
            if self.pacman_pos in self.coins:
                self.coins.remove(self.pacman_pos)
                # Tìm 1 ô trống thay thế
                free_positions = []
                for r in range(1, self.width - 1):
                    for c in range(1, self.height - 1):
                        pos = (r, c)
                        if self.grid[r][c] == 0 and pos != self.pacman_pos and pos not in self.coins:
                            free_positions.append(pos)
                if free_positions:
                    self.coins.add(self.rng.choice(free_positions))
            
        # 5. Đặt Ghost
        if self.randomize_ghost:
            valid_ghost_pos = []
            for r in range(1, self.width - 1):
                for c in range(1, self.height - 1):
                    if self.grid[r][c] == 0 and (r, c) != self.pacman_pos and (r, c) not in self.coins:
                        # Khoảng cách Manhattan >= 3 để tránh Ghost đè/quá gần Pacman
                        dist = abs(r - self.pacman_pos[0]) + abs(c - self.pacman_pos[1])
                        if dist >= 3:
                            valid_ghost_pos.append((r, c))
            if valid_ghost_pos:
                self.ghost_pos = self.rng.choice(valid_ghost_pos)
            else:
                # Fallback nếu không có ô nào xa hơn 3 bước
                fallback_ghost_pos = []
                for r in range(1, self.width - 1):
                    for c in range(1, self.height - 1):
                        if self.grid[r][c] == 0 and (r, c) != self.pacman_pos and (r, c) not in self.coins:
                            fallback_ghost_pos.append((r, c))
                self.ghost_pos = self.rng.choice(fallback_ghost_pos) if fallback_ghost_pos else (self.width - 2, self.height - 2)
        else:
            default_ghost_pos = (self.width - 2, self.height - 2)
            # Kiểm tra xem default_ghost_pos có bị đè bởi Pacman hoặc Coins không
            if default_ghost_pos == self.pacman_pos or default_ghost_pos in self.coins:
                fallback_ghost_pos = []
                for r in range(1, self.width - 1):
                    for c in range(1, self.height - 1):
                        if self.grid[r][c] == 0 and (r, c) != self.pacman_pos and (r, c) not in self.coins:
                            fallback_ghost_pos.append((r, c))
                self.ghost_pos = self.rng.choice(fallback_ghost_pos) if fallback_ghost_pos else default_ghost_pos
            else:
                self.ghost_pos = default_ghost_pos
            
        # 6. Đặt Tường ngẫu nhiên (nếu randomize_walls là True)
        if self.randomize_walls:
            # Thử sinh tường ngẫu nhiên sao cho có đường đi từ Pacman tới tất cả đồng xu và ghost
            success = False
            for _ in range(100): # Thử tối đa 100 lần
                # Tạo lưới tạm
                temp_grid = [[self.grid[r][c] for c in range(self.height)] for r in range(self.width)]
                
                # Tìm các ô trống có thể đặt tường (tránh Pacman, Ghost, Coins)
                free_positions = []
                for r in range(1, self.width - 1):
                    for c in range(1, self.height - 1):
                        pos = (r, c)
                        if pos != self.pacman_pos and pos != self.ghost_pos and pos not in self.coins:
                            free_positions.append(pos)
                
                # Đặt ngẫu nhiên 5 bức tường trong số các ô trống này
                if len(free_positions) >= 5:
                    wall_pos = self.rng.sample(free_positions, 5)
                    for wr, wc in wall_pos:
                        temp_grid[wr][wc] = 1
                
                # Kiểm tra tính liên thông (BFS từ Pacman tới mọi coins)
                reachable = True
                for coin in self.coins:
                    queue = [self.pacman_pos]
                    visited = {self.pacman_pos}
                    found = False
                    while queue:
                        curr = queue.pop(0)
                        if curr == coin:
                            found = True
                            break
                        cr, cc = curr
                        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < self.width and 0 <= nc < self.height and temp_grid[nr][nc] == 0:
                                if (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                    if not found:
                        reachable = False
                        break
                
                if reachable:
                    self.grid = temp_grid
                    success = True
                    break
            
            if not success:
                # Nếu 100 lần thất bại, phục hồi lưới tường tiêu chuẩn và đặt lại Pacman/Ghost/Coins
                self.grid = self._get_default_grid()
                self.coins = self._get_default_coins()
                self.pacman_pos = (1, 1)
                self.ghost_pos = (self.width - 2, self.height - 2)
        self._precompute_shortest_paths()
        return self.state_encoder()

    def _get_valid_neighbors(self, pos: Tuple[int, int]) -> list:
        r, c = pos
        neighbors = []
        # Các hướng di chuyển tương ứng với thứ tự UP, RIGHT, DOWN, LEFT
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.width and 0 <= nc < self.height and self.grid[nr][nc] == 0:
                neighbors.append((nr, nc))
        return neighbors

    def _get_default_grid(self) -> list:
        if self.walkable_width == 7 and self.walkable_height == 7:
            return [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        elif self.walkable_width == 6 and self.walkable_height == 6:
            return [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        elif self.walkable_width == 5 and self.walkable_height == 5:
            return [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        else:
            grid = [[1] * self.height for _ in range(self.width)]
            for r in range(1, self.width - 1):
                for c in range(1, self.height - 1):
                    grid[r][c] = 0
            return grid

    def _get_default_coins(self) -> set:
        if self.walkable_width == 7 and self.walkable_height == 7:
            return {(1, 4), (1, 6), (2, 1), (3, 1), (3, 4)}
        elif self.walkable_width == 6 and self.walkable_height == 6:
            return {(1, 3), (1, 5), (2, 1), (3, 1), (3, 4)}
        elif self.walkable_width == 5 and self.walkable_height == 5:
            return {(1, 3), (1, 4), (2, 1), (3, 1), (3, 2)}
        else:
            return {(1, 2), (1, 3), (2, 1), (2, 2)}

    def _precompute_shortest_paths(self):
        self.shortest_paths = {}
        # Tìm tất cả các ô có thể đi qua
        walkable = []
        for r in range(self.width):
            for c in range(self.height):
                if self.grid[r][c] == 0:
                    walkable.append((r, c))
                    
        directions = [
            ((-1, 0), 'up'),
            ((0, 1), 'right'),
            ((1, 0), 'down'),
            ((0, -1), 'left')
        ]
        
        for start in walkable:
            queue = []
            visited = {start}
            
            # Khởi tạo queue bằng các láng giềng của start
            r, c = start
            for (dr, dc), dir_name in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.width and 0 <= nc < self.height and self.grid[nr][nc] == 0:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), 1, dir_name))
                    self.shortest_paths[(start, (nr, nc))] = (1, dir_name)
            
            self.shortest_paths[(start, start)] = (0, 'none')
            
            while queue:
                curr, dist, first_dir = queue.pop(0)
                cr, cc = curr
                for (dr, dc), _ in directions:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < self.width and 0 <= nc < self.height and self.grid[nr][nc] == 0:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            self.shortest_paths[(start, (nr, nc))] = (dist + 1, first_dir)
                            queue.append(((nr, nc), dist + 1, first_dir))

    def _bfs_distance(self, start: Tuple[int, int], end: Tuple[int, int]) -> int:
        if start == end:
            return 0
        if not hasattr(self, 'shortest_paths'):
            self._precompute_shortest_paths()
        return self.shortest_paths.get((start, end), (9999, 'none'))[0]

    def get_shortest_path_dir(self, start: Tuple[int, int], targets: set) -> str:
        if not targets:
            return 'none'
        if start in targets:
            return 'none'
        if not hasattr(self, 'shortest_paths'):
            self._precompute_shortest_paths()
            
        min_dist = float('inf')
        best_dir = 'none'
        dir_priority = {'up': 0, 'right': 1, 'down': 2, 'left': 3, 'none': 4}
        
        for target in targets:
            path_info = self.shortest_paths.get((start, target))
            if path_info is not None:
                dist, dir_name = path_info
                if dist < min_dist:
                    min_dist = dist
                    best_dir = dir_name
                elif dist == min_dist:
                    # Rẽ nhánh ưu tiên (tie-breaking) giống thuật toán BFS ban đầu
                    if dir_priority[dir_name] < dir_priority[best_dir]:
                        best_dir = dir_name
        return best_dir

    def move_ghost(self) -> None:
        neighbors = self._get_valid_neighbors(self.ghost_pos)
        if not neighbors:
            return
        
        # Ghost Stochastic: ghost_chase_prob đuổi Pacman (shortest path BFS), (1 - ghost_chase_prob) ngẫu nhiên
        if self.rng.random() < self.ghost_chase_prob:
            best_neighbor = None
            min_dist = float('inf')
            for neighbor in neighbors:
                dist = self._bfs_distance(neighbor, self.pacman_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_neighbor = neighbor
            if best_neighbor is not None:
                self.ghost_pos = best_neighbor
            else:
                self.ghost_pos = self.rng.choice(neighbors)
        else:
            self.ghost_pos = self.rng.choice(neighbors)

    def step(self, action: int) -> Tuple[Tuple[int, int, int, int], float, bool, bool, Dict[str, Any]]:
        """
        Thực hiện 1 bước game.
        """
        self.current_step += 1
        
        # 1. Tính vị trí tiếp theo của Pacman dựa trên action
        r, c = self.pacman_pos
        hit_wall = False
        
        if action == ACTION_UP:
            new_pos = (r - 1, c)
        elif action == ACTION_RIGHT:
            new_pos = (r, c + 1)
        elif action == ACTION_DOWN:
            new_pos = (r + 1, c)
        elif action == ACTION_LEFT:
            new_pos = (r, c - 1)
        else:
            new_pos = (r, c)
            
        # Kiểm tra va chạm tường
        if 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height and self.grid[new_pos[0]][new_pos[1]] == 0:
            self.pacman_pos = new_pos
        else:
            hit_wall = True
            
        # Kiểm tra xem Pacman có chủ động đâm vào Ghost không
        caught = (self.pacman_pos == self.ghost_pos)
        ate_coin = False
        ate_all_coins = False
        
        if not caught:
            # Ăn xu
            if self.pacman_pos in self.coins:
                self.coins.remove(self.pacman_pos)
                ate_coin = True
                if len(self.coins) == 0:
                    ate_all_coins = True
                    
        # Nếu game chưa kết thúc, di chuyển Ghost
        if not caught and not ate_all_coins:
            self.move_ghost()
            # Kiểm tra xem Ghost có di chuyển đâm vào Pacman không
            if self.pacman_pos == self.ghost_pos:
                caught = True
                
        # Cập nhật kết thúc
        terminated = False
        if caught:
            reward = self.rewards.get('caught', -50.0)
            terminated = True
        elif ate_all_coins:
            reward = self.rewards.get('win', 30.0)
            terminated = True
        elif ate_coin:
            reward = self.rewards.get('coin', 5.0)
        elif hit_wall:
            reward = self.rewards.get('wall', -5.0)
        else:
            reward = self.rewards.get('step', -1.0)
            
        truncated = self.current_step >= self.max_steps
        
        next_state = self.state_encoder()
        
        info = {
            'pacman_pos': self.pacman_pos,
            'ghost_pos': self.ghost_pos,
            'coins_left': len(self.coins),
            'hit_wall': hit_wall,
            'caught': caught
        }
        
        return next_state, reward, terminated, truncated, info

    def state_encoder(self) -> Tuple[int, int, int, int]:
        """
        Nén trạng thái thực tế (tọa độ thực tế) thành tuple chỉ số số nguyên để tra cứu Q-table.
        """
        # 1. agent_pos_index: Số hóa ô hiện tại của Pacman (r * height + c)
        agent_pos_index = self.pacman_pos[0] * self.height + self.pacman_pos[1]
        
        # 2. ghost_zone_index: Xác định vùng nguy hiểm của ghost (bán kính BFS <= 2)
        pr, pc = self.pacman_pos
        gr, gc = self.ghost_pos
        
        if gr == pr - 1 and gc == pc:
            ghost_zone = 'near_up'
        elif gr == pr and gc == pc + 1:
            ghost_zone = 'near_right'
        elif gr == pr + 1 and gc == pc:
            ghost_zone = 'near_down'
        elif gr == pr and gc == pc - 1:
            ghost_zone = 'near_left'
        else:
            # Nếu không kề cạnh, dùng BFS để quét tầm xa 2 ô
            dist = self._bfs_distance(self.pacman_pos, self.ghost_pos)
            if dist <= 2:
                # Tính toán quyết định cảnh báo 50/50 ở khoảng cách 2 bằng hàm băm đa thức thuần Python
                # Điều này giúp giảm tỷ lệ thắng xuống tầm 70~80% đồng bộ trên mọi kích thước bản đồ
                seed_val = self._seed if self._seed is not None else 42
                state_str = f"{self.pacman_pos[0]}_{self.pacman_pos[1]}_{self.ghost_pos[0]}_{self.ghost_pos[1]}_{self.current_step}_{seed_val}"
                hash_val = 0
                for char in state_str:
                    hash_val = (hash_val * 31 + ord(char)) % 1000000007
                should_warn = (hash_val % 100) < 50
                
                if should_warn:
                    ghost_dir = self.get_shortest_path_dir(self.pacman_pos, {self.ghost_pos})
                    if ghost_dir == 'up':
                        ghost_zone = 'near_up'
                    elif ghost_dir == 'right':
                        ghost_zone = 'near_right'
                    elif ghost_dir == 'down':
                        ghost_zone = 'near_down'
                    elif ghost_dir == 'left':
                        ghost_zone = 'near_left'
                    else:
                        ghost_zone = 'far'
                else:
                    ghost_zone = 'far'
            else:
                ghost_zone = 'far'
        ghost_zone_index = GHOST_ZONE_MAP[ghost_zone]
        
        # 3. nearest_coin_dir_index: Hướng của đồng xu gần nhất
        nearest_coin_dir = self.get_shortest_path_dir(self.pacman_pos, self.coins)
        nearest_coin_dir_index = COIN_DIR_MAP[nearest_coin_dir]
        
        # 4. coin_count_bin_index: Phân nhóm số lượng xu còn lại (low, medium, high)
        coins_left = len(self.coins)
        if coins_left <= 1:
            coin_count_bin = 'low'
        elif coins_left <= 3:
            coin_count_bin = 'medium'
        else:
            coin_count_bin = 'high'
        coin_count_bin_index = COIN_COUNT_MAP[coin_count_bin]
        
        return (agent_pos_index, ghost_zone_index, nearest_coin_dir_index, coin_count_bin_index)

    def state_decoder(self, state: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Giải mã trạng thái nén về dạng dễ đọc.

        Decoder này phục vụ debug/hiển thị; không thể khôi phục đầy đủ trạng thái
        game (ví dụ: tập coin), chỉ trả về các thành phần đã encode.
        """
        agent_pos_index, ghost_zone_index, nearest_coin_dir_index, coin_count_bin_index = state

        if not (0 <= agent_pos_index < (self.width * self.height)):
            raise ValueError(f"agent_pos_index out of range: {agent_pos_index}")

        agent_pos = (agent_pos_index // self.height, agent_pos_index % self.height)
        ghost_zone = INV_GHOST_ZONE_MAP.get(ghost_zone_index, 'unknown')
        nearest_coin_dir = INV_COIN_DIR_MAP.get(nearest_coin_dir_index, 'unknown')
        coin_count_bin = INV_COIN_COUNT_MAP.get(coin_count_bin_index, 'unknown')

        return {
            'agent_pos': agent_pos,
            'ghost_zone': ghost_zone,
            'nearest_coin_dir': nearest_coin_dir,
            'coin_count_bin': coin_count_bin,
        }

    def render(self) -> None:
        """
        Render giao diện dạng văn bản trực quan.
        """
        for r in range(self.width):
            row_str = ""
            for c in range(self.height):
                if (r, c) == self.pacman_pos:
                    row_str += "P "
                elif (r, c) == self.ghost_pos:
                    row_str += "G "
                elif (r, c) in self.coins:
                    row_str += "o "
                elif self.grid[r][c] == 1:
                    row_str += "# "
                else:
                    row_str += ". "
            print(row_str)
        print("-" * 20)

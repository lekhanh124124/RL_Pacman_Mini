import abc
from typing import Tuple, Dict, Any, Optional

class BaseEnv(abc.ABC):
    """
    Lớp môi trường cơ sở (Base Environment) định nghĩa Hợp đồng API (API Contract).
    Tất cả các thành viên trong nhóm tuân thủ interface này để phát triển song song.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed(seed)

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[int, int, int, int]:
        """
        Khởi tạo lại trạng thái của môi trường.
        
        Args:
            seed: Seed ngẫu nhiên để tái lập kết quả.
            
        Returns:
            Trạng thái nén ban đầu dưới dạng tuple: 
            (agent_pos, ghost_zone, nearest_coin_dir, coin_count_bin)
        """
        pass

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[Tuple[int, int, int, int], float, bool, bool, Dict[str, Any]]:
        """
        Thực hiện một hành động của Pacman, cập nhật môi trường (bao gồm cả Ghost di chuyển).
        
        Args:
            action: Hành động của Pacman (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
            
        Returns:
            next_state: Trạng thái nén kế tiếp (tuple)
            reward: Điểm thưởng/phạt tức thời (float)
            terminated: Trạng thái kết thúc do thắng (ăn hết xu) hoặc thua (bị ma bắt) (bool)
            truncated: Trạng thái bị cắt ngắn (ví dụ: quá số bước tối đa) (bool)
            info: Thông tin phụ để debug (tọa độ thực tế, số xu còn lại, v.v.) (dict)
        """
        pass

    @abc.abstractmethod
    def render(self) -> None:
        """
        Hiển thị trạng thái hiện tại của bàn cờ (dạng text hoặc đồ họa).
        """
        pass

    @abc.abstractmethod
    def seed(self, seed: Optional[int] = None) -> None:
        """
        Thiết lập seed ngẫu nhiên cho môi trường để đảm bảo tính tái lập.
        """
        pass

    @abc.abstractmethod
    def state_encoder(self) -> Tuple[int, int, int, int]:
        """
        Mã hóa/Nén trạng thái thực tế của game thành dạng Tuple nén dùng cho Q-table.
        Format: (agent_pos, ghost_zone, nearest_coin_dir, coin_count_bin)
        """
        pass

    @abc.abstractmethod
    def state_decoder(self, state: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Giải mã trạng thái nén về dạng dễ đọc để debug/hiển thị.

        Lưu ý: Vì state là dạng nén, decoder không thể khôi phục đầy đủ toàn bộ
        trạng thái gốc (ví dụ: vị trí tất cả coin). Decoder chỉ trả về các thành
        phần đã được encode.

        Args:
            state: Trạng thái nén (agent_pos, ghost_zone, nearest_coin_dir, coin_count_bin)

        Returns:
            dict gồm: agent_pos (row,col), ghost_zone (str), nearest_coin_dir (str), coin_count_bin (str)
        """
        pass

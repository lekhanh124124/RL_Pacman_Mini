import random
from typing import Tuple

class RandomAgent:
    """
    Random Agent di chuyển ngẫu nhiên hoàn toàn.
    Dùng để làm baseline cơ sở thấp nhất để đối chứng.
    """
    def __init__(self, action_space_size: int = 4):
        self.action_space_size = action_space_size

    def choose_action(self, state: Tuple[int, int, int, int], evaluation: bool = False) -> int:
        """
        Chọn ngẫu nhiên một hành động mà không cần quan tâm đến state.
        """
        return random.randint(0, self.action_space_size - 1)

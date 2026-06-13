import os

import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(results: dict, window_size: int = 10, save_path: str = None):
    """
    Vẽ biểu đồ so sánh đường cong học tập (Learning Curves) của các Agent.
    results format: {
        'Q-Learning': [rewards_over_episodes],
        'SARSA': [rewards_over_episodes],
        'Heuristic': [rewards_over_episodes],
        'Random': [rewards_over_episodes]
    }
    """
    plt.figure(figsize=(10, 6))
    
    for agent_name, rewards in results.items():
        episodes = len(rewards)
        # Tính trung bình trượt (Moving Average) để đồ thị mịn và dễ đọc hơn
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        x_axis = np.arange(window_size - 1, episodes)
        
        plt.plot(x_axis, smoothed_rewards, label=agent_name, linewidth=2)
        
    plt.title("So Sánh Đường Cong Học Tập (Pacman Mini)", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes (Ván chơi)", fontsize=12)
    plt.ylabel("Moving Average Reward (Điểm thưởng trung bình trượt)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    if save_path:
        # Tự động tạo thư mục cha nếu chưa có
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ thành công tại: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Dữ liệu giả lập để test khung sườn vẽ đồ thị
    import os
    dummy_results = {
        'Q-Learning': np.cumsum(np.random.normal(0.1, 1.0, 500)) - 100,
        'SARSA': np.cumsum(np.random.normal(0.08, 1.0, 500)) - 100,
        'Heuristic': [-20] * 500,
        'Random': [-80] * 500
    }
    
    # Thử vẽ và lưu vào thư mục reports/figures/
    save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_save_path = os.path.join(save_dir, "reports", "figures", "test_learning_curve.png")
    plot_learning_curves(dummy_results, window_size=20, save_path=test_save_path)

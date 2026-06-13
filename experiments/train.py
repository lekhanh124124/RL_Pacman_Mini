import sys
import os
import numpy as np
import json
import yaml

# Thêm thư mục src vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import PacmanMiniEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.double_q_learning import DoubleQLearningAgent

def train_all_agents():
    # Đọc cấu hình từ configs.yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    seeds = config["training"].get("seeds", [10, 24, 42, 99, 123, 456, 789, 1000, 2026, 9999])
    num_train_episodes = config["training"]["num_episodes"]
    
    q_cfg = config["q_learning"]
    s_cfg = config["sarsa"]
    dq_cfg = config.get("double_q_learning", q_cfg)
    
    ghost_chase_prob = config["env"].get("ghost_chase_prob", 0.3)
    randomize_pacman = config["env"].get("randomize_pacman", False)
    randomize_ghost = config["env"].get("randomize_ghost", False)
    randomize_coins = config["env"].get("randomize_coins", False)
    randomize_walls = config["env"].get("randomize_walls", False)
    max_steps = config["env"].get("max_steps", 100)
    num_coins = config["env"].get("num_coins", 5)
    rewards = config["env"].get("rewards", None)

    # Đảm bảo tồn tại các thư mục lưu trữ
    os.makedirs("dashboard/policies", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    all_history_data = {}
    map_sizes = [5, 6, 7]

    for map_size in map_sizes:
        width = map_size
        height = map_size
        
        print("\n" + "="*60)
        print(f" HUẤN LUYỆN BẢN ĐỒ KÍCH THƯỚC {width}x{height} (WALKABLE)")
        print("="*60)
        
        q_learning_train_history = np.zeros(num_train_episodes)
        sarsa_train_history = np.zeros(num_train_episodes)
        double_q_train_history = np.zeros(num_train_episodes)

        for seed_idx, seed in enumerate(seeds):
            print(f"\n--- [SEED {seed_idx+1}/{len(seeds)} - ID: {seed}] ---")
            
            # Khởi tạo môi trường
            env = PacmanMiniEnv(
                width=width,
                height=height,
                seed=seed,
                ghost_chase_prob=ghost_chase_prob,
                randomize_pacman=randomize_pacman,
                randomize_ghost=randomize_ghost,
                randomize_coins=randomize_coins,
                randomize_walls=randomize_walls,
                max_steps=max_steps,
                num_coins=num_coins,
                rewards=rewards
            )
            
            # 1. Huấn luyện Q-Learning
            print(" -> Huấn luyện Q-Learning Agent...")
            q_agent = QLearningAgent(
                alpha=q_cfg["alpha"],
                gamma=q_cfg["gamma"],
                epsilon=q_cfg["epsilon_start"],
                epsilon_decay=q_cfg["epsilon_decay"],
                epsilon_min=q_cfg["epsilon_min"]
            )
            for ep in range(num_train_episodes):
                state = env.reset(seed=seed + ep)
                total_r = 0
                term = False
                trunc = False
                while not (term or trunc):
                    action = q_agent.choose_action(state)
                    next_state, reward, term, trunc, _ = env.step(action)
                    q_agent.update(state, action, reward, next_state, term)
                    state = next_state
                    total_r += reward
                q_agent.decay_epsilon()
                q_learning_train_history[ep] += total_r
                
            # Lưu Q-table cho seed này
            q_agent.save_policy(f"dashboard/policies/q_learning_seed_{seed}_{width}x{height}.json")
            if seed_idx == 0:
                q_agent.save_policy(f"dashboard/q_learning_policy_{width}x{height}.json")

            # 2. Huấn luyện SARSA
            print(" -> Huấn luyện SARSA Agent...")
            sarsa_agent = SarsaAgent(
                alpha=s_cfg["alpha"],
                gamma=s_cfg["gamma"],
                epsilon=s_cfg["epsilon_start"],
                epsilon_decay=s_cfg["epsilon_decay"],
                epsilon_min=s_cfg["epsilon_min"]
            )
            for ep in range(num_train_episodes):
                state = env.reset(seed=seed + ep)
                action = sarsa_agent.choose_action(state)
                total_r = 0
                term = False
                trunc = False
                while not (term or trunc):
                    next_state, reward, term, trunc, _ = env.step(action)
                    next_action = sarsa_agent.choose_action(next_state)
                    sarsa_agent.update(state, action, reward, next_state, next_action, term)
                    state = next_state
                    action = next_action
                    total_r += reward
                sarsa_agent.decay_epsilon()
                sarsa_train_history[ep] += total_r
                
            # Lưu Q-table cho seed này
            sarsa_agent.save_policy(f"dashboard/policies/sarsa_seed_{seed}_{width}x{height}.json")
            if seed_idx == 0:
                sarsa_agent.save_policy(f"dashboard/sarsa_policy_{width}x{height}.json")

            # 3. Huấn luyện Double Q-Learning
            print(" -> Huấn luyện Double Q-Learning Agent...")
            double_q_agent = DoubleQLearningAgent(
                alpha=dq_cfg["alpha"],
                gamma=dq_cfg["gamma"],
                epsilon=dq_cfg["epsilon_start"],
                epsilon_decay=dq_cfg["epsilon_decay"],
                epsilon_min=dq_cfg["epsilon_min"]
            )
            for ep in range(num_train_episodes):
                state = env.reset(seed=seed + ep)
                total_r = 0
                term = False
                trunc = False
                while not (term or trunc):
                    action = double_q_agent.choose_action(state)
                    next_state, reward, term, trunc, _ = env.step(action)
                    double_q_agent.update(state, action, reward, next_state, term)
                    state = next_state
                    total_r += reward
                double_q_agent.decay_epsilon()
                double_q_train_history[ep] += total_r
                
            # Lưu Q-table cho seed này
            double_q_agent.save_policy(f"dashboard/policies/double_q_seed_{seed}_{width}x{height}.json")
            if seed_idx == 0:
                double_q_agent.save_policy(f"dashboard/double_q_policy_{width}x{height}.json")

        # Tính trung bình lịch sử huấn luyện qua các seeds
        q_learning_train_history /= len(seeds)
        sarsa_train_history /= len(seeds)
        double_q_train_history /= len(seeds)

        all_history_data[f"{width}x{height}"] = {
            "q_learning": q_learning_train_history.tolist(),
            "sarsa": sarsa_train_history.tolist(),
            "double_q_learning": double_q_train_history.tolist()
        }

    # Ghi toàn bộ lịch sử huấn luyện ra tệp JSON
    with open("reports/training_history.json", "w", encoding="utf-8") as f:
        json.dump(all_history_data, f, indent=4)
        
    print("\n🎉 TIẾN TRÌNH HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG CHO CẢ 3 BẢN ĐỒ!")
    print("-> Đã lưu các chính sách Q-table tại: dashboard/policies/")
    print("-> Đã lưu lịch sử huấn luyện tại: reports/training_history.json")

if __name__ == "__main__":
    train_all_agents()

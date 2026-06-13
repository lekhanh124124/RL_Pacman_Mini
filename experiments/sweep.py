import sys
import os
import numpy as np
import yaml

# Thêm thư mục src vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import PacmanMiniEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.double_q_learning import DoubleQLearningAgent

def evaluate_hyperparameters(agent_type: str, alpha: float, gamma: float, episodes: int = 500, seeds: list = None) -> float:
    """
    Huấn luyện và đánh giá một thuật toán cụ thể (agent_type) với tổ hợp siêu tham số (alpha, gamma).
    """
    if seeds is None:
        seeds = [42, 123, 999]  # 3 seeds đại diện để tối ưu tốc độ chạy
        
    all_success_rates = []
    
    for seed in seeds:
        # Cố định seed toàn cục để đảm bảo tính tái lập 100% của tác nhân (tie-breaking, exploration)
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        env = PacmanMiniEnv(seed=seed)
        
        # Khởi tạo agent tương ứng
        if agent_type == 'q_learning':
            agent = QLearningAgent(action_space_size=4, alpha=alpha, gamma=gamma, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)
        elif agent_type == 'sarsa':
            agent = SarsaAgent(action_space_size=4, alpha=alpha, gamma=gamma, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)
        elif agent_type == 'double_q_learning':
            agent = DoubleQLearningAgent(action_space_size=4, alpha=alpha, gamma=gamma, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        success_rates = []
        for episode in range(episodes):
            state = env.reset(seed=seed + episode)
            terminated = False
            truncated = False
            
            # Lựa chọn bước đầu tiên cho thuật toán On-Policy (SARSA)
            if agent_type == 'sarsa':
                action = agent.choose_action(state)
                
            while not (terminated or truncated):
                if agent_type == 'sarsa':
                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_action = agent.choose_action(next_state)
                    agent.update(state, action, reward, next_state, next_action, terminated)
                    state = next_state
                    action = next_action
                else:
                    action = agent.choose_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    agent.update(state, action, reward, next_state, terminated)
                    state = next_state
                    
            agent.decay_epsilon()
            # Thắng nếu ăn hết xu
            is_success = terminated and (info.get('coins_left', 1) == 0)
            success_rates.append(1.0 if is_success else 0.0)
            
        # Lấy trung bình tỷ lệ thắng của 100 ván cuối cùng (giai đoạn hội tụ)
        all_success_rates.append(np.mean(success_rates[-100:]))
        
    return float(np.mean(all_success_rates))

def run_parameter_sweep():
    """
    Quét qua các tổ hợp siêu tham số để tìm cấu hình tối ưu chung tốt nhất cho cả 3 thuật toán RL.
    """
    # Đọc cấu hình từ configs.yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        sweep_cfg = config.get("sweep", {})
    except Exception as e:
        print(f"Warning: Không thể đọc configs.yaml ({e}), sử dụng tham số mặc định.")
        sweep_cfg = {}

    alphas = sweep_cfg.get("alphas", [0.01, 0.05, 0.1, 0.2])
    gammas = sweep_cfg.get("gammas", [0.85, 0.90, 0.95])
    episodes = sweep_cfg.get("num_episodes", 500)
    seeds = sweep_cfg.get("seeds", [42, 123, 999])

    print("=============================================================")
    print(" BẮT ĐẦU QUÉT SIÊU THAM SỐ ĐA THUẬT TOÁN (RL GRID SWEEP)")
    print("=============================================================")
    
    best_avg_rate = -1.0
    best_config = None
    results_table = []
    
    total_runs = len(alphas) * len(gammas)
    run_idx = 1
    
    for alpha in alphas:
        for gamma in gammas:
            print(f"[{run_idx}/{total_runs}] Đang đánh giá tổ hợp: alpha={alpha}, gamma={gamma} ...")
            
            # Đánh giá trên cả 3 thuật toán
            q_rate = evaluate_hyperparameters('q_learning', alpha, gamma, episodes=episodes, seeds=seeds)
            sarsa_rate = evaluate_hyperparameters('sarsa', alpha, gamma, episodes=episodes, seeds=seeds)
            double_q_rate = evaluate_hyperparameters('double_q_learning', alpha, gamma, episodes=episodes, seeds=seeds)
            
            avg_rate = (q_rate + sarsa_rate + double_q_rate) / 3.0
            
            print(f"   -> Q-Learning: {q_rate*100:.1f}% | SARSA: {sarsa_rate*100:.1f}% | Double Q: {double_q_rate*100:.1f}%")
            print(f"   -> Hiệu năng TRUNG BÌNH CHUNG: {avg_rate*100:.1f}%")
            
            results_table.append({
                'alpha': alpha,
                'gamma': gamma,
                'q_rate': q_rate,
                'sarsa_rate': sarsa_rate,
                'double_q_rate': double_q_rate,
                'avg_rate': avg_rate
            })
            
            if avg_rate > best_avg_rate:
                best_avg_rate = avg_rate
                best_config = (alpha, gamma)
                
            run_idx += 1
            
    print("\n=============================================================")
    print(" KẾT QUẢ QUÉT SIÊU THAM SỐ TOÀN DIỆN (SUMMARY)")
    print("=============================================================")
    print(f"{'Alpha':<8}{'Gamma':<8}{'Q-Learn':<10}{'SARSA':<10}{'Double Q':<10}{'TRUNG BÌNH':<12}")
    print("-" * 60)
    for res in results_table:
        print(f"{res['alpha']:<8}{res['gamma']:<8}{res['q_rate']*100:.1f}%     {res['sarsa_rate']*100:.1f}%     {res['double_q_rate']*100:.1f}%     {res['avg_rate']*100:.1f}%")
    print("-" * 60)
    print(f"🎉 Cấu hình tối ưu chung tốt nhất: alpha={best_config[0]}, gamma={best_config[1]} (Trung bình: {best_avg_rate*100:.1f}%)")
    print("=============================================================")
    
    # Cập nhật cấu hình tối ưu vào configs.yaml nếu cần thiết
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Cập nhật siêu tham số tối ưu vào file cấu hình cho cả ba phần
            if config_data:
                config_data['q_learning']['alpha'] = best_config[0]
                config_data['q_learning']['gamma'] = best_config[1]
                config_data['sarsa']['alpha'] = best_config[0]
                config_data['sarsa']['gamma'] = best_config[1]
                if 'double_q_learning' in config_data:
                    config_data['double_q_learning']['alpha'] = best_config[0]
                    config_data['double_q_learning']['gamma'] = best_config[1]
                
                class CustomDumper(yaml.SafeDumper):
                    def represent_sequence(self, tag, sequence, flow_style=None):
                        if all(isinstance(x, (int, float, str)) for x in sequence):
                            flow_style = True
                        return super().represent_sequence(tag, sequence, flow_style)

                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, Dumper=CustomDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
                print(f"Đã tự động cập nhật cấu hình tối ưu vào file {os.path.basename(config_path)}!")
        except Exception as e:
            print(f"Lưu ý: Không thể tự động ghi đè file cấu hình: {e}")

if __name__ == "__main__":
    run_parameter_sweep()

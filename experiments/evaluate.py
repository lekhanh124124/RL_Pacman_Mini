import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import yaml

# Thêm thư mục src vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import PacmanMiniEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.double_q_learning import DoubleQLearningAgent

def evaluate_all_agents():
    # Đọc cấu hình từ configs.yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    seeds = config["training"].get("seeds", [10, 24, 42, 99, 123, 456, 789, 1000, 2026, 9999])
    num_eval_episodes = config.get("evaluation", {}).get("num_episodes", 100)
    window_size_val = config.get("evaluation", {}).get("window_size", 20)
    
    ghost_chase_prob = config["env"].get("ghost_chase_prob", 0.3)
    randomize_pacman = config["env"].get("randomize_pacman", False)
    randomize_ghost = config["env"].get("randomize_ghost", False)
    randomize_coins = config["env"].get("randomize_coins", False)
    randomize_walls = config["env"].get("randomize_walls", False)
    max_steps = config["env"].get("max_steps", 100)
    num_coins = config["env"].get("num_coins", 5)
    rewards = config["env"].get("rewards", None)

    # Đảm bảo tồn tại các thư mục lưu trữ
    os.makedirs("src/reports/figures", exist_ok=True)

    summary_data = {}
    map_sizes = [5, 6, 7]

    for map_size in map_sizes:
        width = map_size
        height = map_size
        
        print("\n" + "="*60)
        print(f" TIẾN TRÌNH ĐÁNH GIÁ BẢN ĐỒ KÍCH THƯỚC {width}x{height} (10 SEEDS)")
        print("="*60)

        # Cấu trúc lưu trữ kết quả đánh giá của từng seed
        results = {
            'Random': {'success_rates': [], 'rewards': [], 'steps': [], 'steps_on_win': [], 'deaths': [], 'wall_hits': []},
            'Heuristic': {'success_rates': [], 'rewards': [], 'steps': [], 'steps_on_win': [], 'deaths': [], 'wall_hits': []},
            'Q-Learning': {'success_rates': [], 'rewards': [], 'steps': [], 'steps_on_win': [], 'deaths': [], 'wall_hits': []},
            'SARSA': {'success_rates': [], 'rewards': [], 'steps': [], 'steps_on_win': [], 'deaths': [], 'wall_hits': []},
            'Double Q-Learning': {'success_rates': [], 'rewards': [], 'steps': [], 'steps_on_win': [], 'deaths': [], 'wall_hits': []}
        }

        # Lưu trữ tất cả số bước đi khi thắng để vẽ Boxplot
        all_agent_steps_on_win = {
            'Random': [],
            'Heuristic': [],
            'Q-Learning': [],
            'SARSA': [],
            'Double Q-Learning': []
        }

        for seed_idx, seed in enumerate(seeds):
            print(f"\n--- [SEED {seed_idx+1}/{len(seeds)} - ID: {seed}] ---")
            
            # Khởi tạo môi trường đánh giá
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
            
            # Khởi tạo các Agent và nạp Q-tables tương ứng từ đĩa
            q_agent = QLearningAgent(epsilon=0.0)
            s_agent = SarsaAgent(epsilon=0.0)
            dq_agent = DoubleQLearningAgent(epsilon=0.0)
            
            q_policy_path = f"src/dashboard/policies/q_learning_seed_{seed}_{width}x{height}.json"
            s_policy_path = f"src/dashboard/policies/sarsa_seed_{seed}_{width}x{height}.json"
            dq_policy_path = f"src/dashboard/policies/double_q_seed_{seed}_{width}x{height}.json"
            
            if os.path.exists(q_policy_path):
                q_agent.load_policy(q_policy_path)
            else:
                print(f" Warning: Không tìm thấy Q-Learning policy cho seed {seed} map {width}x{height}.")
                
            if os.path.exists(s_policy_path):
                s_agent.load_policy(s_policy_path)
            else:
                print(f" Warning: Không tìm thấy SARSA policy cho seed {seed} map {width}x{height}.")
                
            if os.path.exists(dq_policy_path):
                dq_agent.load_policy(dq_policy_path)
            else:
                print(f" Warning: Không tìm thấy Double Q-Learning policy cho seed {seed} map {width}x{height}.")

            agents_to_eval = {
                'Random': RandomAgent(),
                'Heuristic': HeuristicAgent(),
                'Q-Learning': q_agent,
                'SARSA': s_agent,
                'Double Q-Learning': dq_agent
            }
            
            for agent_name, agent in agents_to_eval.items():
                ep_rewards = []
                ep_steps = []
                ep_steps_on_win = []
                ep_wall_hits = []
                successes = 0
                deaths = 0
                
                for ep in range(num_eval_episodes):
                    state = env.reset(seed=seed + 1000 + ep)
                    total_r = 0
                    steps = 0
                    wall_hits = 0
                    term = False
                    trunc = False
                    
                    while not (term or trunc):
                        action = agent.choose_action(state, evaluation=True)
                        next_state, reward, term, trunc, info = env.step(action)
                        state = next_state
                        total_r += reward
                        steps += 1
                        if info.get('hit_wall', False):
                            wall_hits += 1
                    
                    ep_rewards.append(total_r)
                    ep_steps.append(steps)
                    ep_wall_hits.append(wall_hits)
                    
                    if term and info.get('coins_left', 1) == 0:
                        successes += 1
                        ep_steps_on_win.append(steps)
                    elif term and info.get('caught', False):
                        deaths += 1
                        
                success_rate = successes / num_eval_episodes
                death_rate = deaths / num_eval_episodes
                mean_steps_all = float(np.mean(ep_steps))
                mean_steps_on_win = float(np.mean(ep_steps_on_win)) if ep_steps_on_win else float('nan')
                mean_wall_hits = float(np.mean(ep_wall_hits))

                results[agent_name]['success_rates'].append(success_rate)
                results[agent_name]['rewards'].append(float(np.mean(ep_rewards)))
                results[agent_name]['steps'].append(mean_steps_all)
                results[agent_name]['steps_on_win'].append(mean_steps_on_win)
                results[agent_name]['deaths'].append(death_rate)
                results[agent_name]['wall_hits'].append(mean_wall_hits)
                all_agent_steps_on_win[agent_name].extend(ep_steps_on_win)

                steps_on_win_str = f"{mean_steps_on_win:.2f}" if np.isfinite(mean_steps_on_win) else "n/a"
                print(
                    f"    * {agent_name:<18} -> Win: {success_rate*100:5.1f}%, "
                    f"Steps(all): {mean_steps_all:.2f}, Wall Hits: {mean_wall_hits:.2f}"
                )
                
        # --- TÍNH TOÁN THỐNG KÊ (MEAN ± STD) ---
        print("\n" + "="*60)
        print(f" KẾT QUẢ ĐÁNH GIÁ TOÀN DIỆN {width}x{height} (MEAN ± STD)")
        print("="*60)
        
        size_key = f"{width}x{height}"
        summary_data[size_key] = {}
        for agent_name, metrics in results.items():
            mean_success = np.mean(metrics['success_rates']) * 100
            std_success = np.std(metrics['success_rates']) * 100
            
            mean_reward = np.mean(metrics['rewards'])
            std_reward = np.std(metrics['rewards'])
            
            mean_steps = np.mean(metrics['steps'])
            std_steps = np.std(metrics['steps'])

            mean_steps_on_win = np.nanmean(metrics['steps_on_win'])
            std_steps_on_win = np.nanstd(metrics['steps_on_win'])
            
            mean_deaths = np.mean(metrics['deaths']) * 100
            std_deaths = np.std(metrics['deaths']) * 100

            mean_wall_hits = np.mean(metrics['wall_hits'])
            std_wall_hits = np.std(metrics['wall_hits'])
            
            summary_data[size_key][agent_name] = {
                'success_rate': f"{mean_success:.1f}% ± {std_success:.1f}%",
                'reward': f"{mean_reward:.1f} ± {std_reward:.1f}",
                'steps': f"{mean_steps:.2f} ± {std_steps:.2f}",
                'steps_on_win': (
                    f"{mean_steps_on_win:.2f} ± {std_steps_on_win:.2f}" if np.isfinite(mean_steps_on_win) else "n/a"
                ),
                'death_rate': f"{mean_deaths:.1f}% ± {std_deaths:.1f}%",
                'wall_hits': f"{mean_wall_hits:.2f} ± {std_wall_hits:.2f}",
                'raw': {
                    'success_mean': mean_success, 'success_std': std_success,
                    'reward_mean': mean_reward, 'reward_std': std_reward,
                    'steps_mean': mean_steps, 'steps_std': std_steps,
                    'steps_on_win_mean': mean_steps_on_win, 'steps_on_win_std': std_steps_on_win,
                    'death_mean': mean_deaths, 'death_std': std_deaths,
                    'wall_hits_mean': mean_wall_hits, 'wall_hits_std': std_wall_hits
                }
            }
            
            print(f"\n🔹 Agent: {agent_name}")
            print(f"   - Tỷ lệ thành công:  {summary_data[size_key][agent_name]['success_rate']}")
            print(f"   - Reward tích lũy:   {summary_data[size_key][agent_name]['reward']}")
            print(f"   - Số bước đi TB:     {summary_data[size_key][agent_name]['steps']}")
            print(f"   - Số bước khi THẮNG: {summary_data[size_key][agent_name]['steps_on_win']}")
            print(f"   - Tỷ lệ bị Ghost bắt:{summary_data[size_key][agent_name]['death_rate']}")
            print(f"   - Số lần đâm tường:  {summary_data[size_key][agent_name]['wall_hits']}")

        # --- VẼ CÁC BIỂU ĐỒ SO SÁNH CHO TỪNG KÍCH THƯỚC MAP ---
        print(f"\n-> Đang vẽ đồ thị cho map {width}x{height}...")
        
        # 1. Đồ thị Learning Curves từ training_history.json
        history_path = "src/reports/training_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                
            size_history = history_data.get(size_key, {})
            if size_history:
                q_history = size_history["q_learning"]
                s_history = size_history["sarsa"]
                dq_history = size_history["double_q_learning"]
                num_ep = len(q_history)
                
                plt.figure(figsize=(10, 6))
                window_size = window_size_val
                
                def moving_avg(data, w):
                    return np.convolve(data, np.ones(w)/w, mode='valid')
                    
                plt.plot(np.arange(window_size-1, num_ep), moving_avg(q_history, window_size), label='Q-Learning', linewidth=2.5, color='#1f77b4')
                plt.plot(np.arange(window_size-1, num_ep), moving_avg(s_history, window_size), label='SARSA', linewidth=2.5, color='#ff7f0e')
                plt.plot(np.arange(window_size-1, num_ep), moving_avg(dq_history, window_size), label='Double Q-Learning', linewidth=2.5, color='#2ca02c')
                
                plt.title(f"Đường Cong Học Tập Trung Bình - Bản đồ {width}x{height}", fontsize=14, fontweight='bold', pad=15)
                plt.xlabel("Episodes (Ván chơi)", fontsize=12)
                plt.ylabel(f"Reward Trung bình Trượt (Window={window_size})", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.legend(fontsize=11)
                plt.savefig(f"src/reports/figures/learning_curves_{width}x{height}.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   * Lưu thành công: src/reports/figures/learning_curves_{width}x{height}.png")
                
        # 2. Đồ thị Bar Chart so sánh Tỷ lệ thắng & Tỷ lệ chết
        agents = list(results.keys())
        success_means = [summary_data[size_key][a]['raw']['success_mean'] for a in agents]
        success_stds = [summary_data[size_key][a]['raw']['success_std'] for a in agents]
        
        death_means = [summary_data[size_key][a]['raw']['death_mean'] for a in agents]
        death_stds = [summary_data[size_key][a]['raw']['death_std'] for a in agents]
        
        x = np.arange(len(agents))
        width_bar = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width_bar/2, success_means, width_bar, yerr=success_stds, label='Tỷ lệ Thắng (Ăn sạch xu)', color='#2ca02c', capsize=5, alpha=0.85)
        ax.bar(x + width_bar/2, death_means, width_bar, yerr=death_stds, label='Tỷ lệ Thua (Bị Ghost bắt)', color='#d62728', capsize=5, alpha=0.85)
        
        ax.set_ylabel('Phần trăm (%)', fontsize=12)
        ax.set_title(f'So Sánh Tỷ Lệ Thắng và Tỷ Lệ Bị Ghost Bắt - Bản đồ {width}x{height} (Mean ± Std)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=15, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"src/reports/figures/success_death_comparison_{width}x{height}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   * Lưu thành công: src/reports/figures/success_death_comparison_{width}x{height}.png")

        # 3. Đồ thị Bar Chart so sánh Số bước đi trung bình
        step_means = [summary_data[size_key][a]['raw']['steps_mean'] for a in agents]
        step_stds = [summary_data[size_key][a]['raw']['steps_std'] for a in agents]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(agents, step_means, yerr=step_stds, color='#9467bd', capsize=5, alpha=0.85, width=0.5)
        ax.set_ylabel('Số bước đi TB', fontsize=12)
        ax.set_title(f'So Sánh Số Bước Đi Trung Bình của Một Episode - Bản đồ {width}x{height} (Mean ± Std)', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=15, fontsize=11)
        plt.tight_layout()
        plt.savefig(f"src/reports/figures/steps_comparison_{width}x{height}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   * Lưu thành công: src/reports/figures/steps_comparison_{width}x{height}.png")

        # 4. Đồ thị Bar Chart so sánh Số lần đâm tường trung bình
        wall_hits_means = [summary_data[size_key][a]['raw']['wall_hits_mean'] for a in agents]
        wall_hits_stds = [summary_data[size_key][a]['raw']['wall_hits_std'] for a in agents]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(agents, wall_hits_means, yerr=wall_hits_stds, color='#ff7f0e', capsize=5, alpha=0.85, width=0.5)
        ax.set_ylabel('Số lần đâm tường TB', fontsize=12)
        ax.set_title(f'So Sánh Số Lần Đâm Tường Trung Bình (Vi phạm ràng buộc) - Bản đồ {width}x{height} (Mean ± Std)', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=15, fontsize=11)
        plt.tight_layout()
        plt.savefig(f"src/reports/figures/wall_hits_comparison_{width}x{height}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   * Lưu thành công: src/reports/figures/wall_hits_comparison_{width}x{height}.png")

        # 5. Đồ thị Boxplot phân bổ số bước đi khi thắng (Path Efficiency)
        fig, ax = plt.subplots(figsize=(9, 6))
        boxplot_data = [all_agent_steps_on_win[a] for a in agents]
        
        # Để tránh lỗi labels trong matplotlib cũ/mới, sử dụng set_xticklabels
        ax.boxplot(boxplot_data)
        ax.set_xticklabels(agents, rotation=15, fontsize=11)
        ax.set_ylabel('Số bước đi (khi thắng)', fontsize=12)
        ax.set_title(f'Phân Bố Số Bước Đi Khi Chiến Thắng - Bản đồ {width}x{height}', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"src/reports/figures/steps_boxplot_{width}x{height}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   * Lưu thành công: src/reports/figures/steps_boxplot_{width}x{height}.png")

        # 6. Đồ thị Radar Chart so sánh toàn diện đa chiều
        categories = ['Tỷ lệ Thắng (%)', 'Tỷ lệ Sống sót (%)', 'Tránh đâm tường (%)', 'Hiệu suất bước đi (%)']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        plt.xticks(angles[:-1], categories, fontsize=11, fontweight='bold')
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
        plt.ylim(0, 100)
        
        agent_colors = {
            'Random': '#7f7f7f',
            'Heuristic': '#bcbd22',
            'Q-Learning': '#1f77b4',
            'SARSA': '#ff7f0e',
            'Double Q-Learning': '#2ca02c'
        }
        
        for agent_name in agents:
            raw_data = summary_data[size_key][agent_name]['raw']
            
            win_rate = raw_data['success_mean']
            survival_rate = 100.0 - raw_data['death_mean']
            wall_avoidance = max(0.0, 100.0 * (1.0 - raw_data['wall_hits_mean'] / 10.0))
            path_efficiency = max(0.0, 100.0 * (1.0 - raw_data['steps_mean'] / max_steps))
            
            values = [win_rate, survival_rate, wall_avoidance, path_efficiency]
            values += values[:1]
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=agent_name, color=agent_colors.get(agent_name))
            ax.fill(angles, values, color=agent_colors.get(agent_name), alpha=0.1)
            
        plt.title(f'So Sánh Toàn Diện Đa Chiều (Radar Chart) - Bản đồ {width}x{height}', fontsize=14, fontweight='bold', pad=25)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        plt.tight_layout()
        plt.savefig(f"src/reports/figures/radar_comparison_{width}x{height}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   * Lưu thành công: src/reports/figures/radar_comparison_{width}x{height}.png")

    # Thêm metadata vào tệp kết quả
    summary_data['metadata'] = {
        'num_seeds': len(seeds),
        'num_eval_episodes': num_eval_episodes
    }
    
    # Ghi kết quả so sánh ra JSON để dashboard sử dụng
    with open("src/reports/results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print("\n-> Đã lưu toàn bộ kết quả đánh giá tại: src/reports/results_summary.json")
    print("\n=============================================================")
    print(" TIẾN TRÌNH ĐÁNH GIÁ HOÀN TẤT THÀNH CÔNG CHO CẢ 3 BẢN ĐỒ!")
    print("=============================================================")

if __name__ == "__main__":
    evaluate_all_agents()

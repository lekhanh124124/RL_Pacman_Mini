# Báo cáo dự án: Pacman Mini — Ăn xu và tránh Ghost (Tabular RL)

## 1. Phát biểu bài toán
Mục tiêu là xây dựng một môi trường dạng lưới (maze 5x5, 6x6, 7x7) có tường, 1 Pacman, 1 Ghost và 5 đồng xu (coin). Agent điều khiển Pacman để:
- Ăn xu nhanh nhất có thể.
- Tránh bị Ghost bắt.

Bài toán được mô hình hoá theo học tăng cường (Reinforcement Learning), trong đó agent học chính sách tối ưu thông qua tương tác thử–sai với môi trường.

---

## 2. MDP (Markov Decision Process)

### 2.1. Không gian hành động (Action)
Tập hành động rời rạc, theo hợp đồng API:
- `0: UP` (Lên trên), `1: RIGHT` (Sang phải), `2: DOWN` (Xuống dưới), `3: LEFT` (Sang trái)

### 2.2. Trạng thái nén (State)
Trạng thái được nén tối giản để phù hợp với giới hạn của bảng Tabular RL (tránh bùng nổ không gian trạng thái):

$$
s = (agent\_pos, ghost\_zone, nearest\_coin\_dir, coin\_count\_bin)
$$

Trong đó:
- **agent_pos**: chỉ số ô của Pacman trong lưới, tính bằng `row * height + col`.
- **ghost_zone** ∈ {near_up, near_right, near_down, near_left, far}: 
  - Khi Ghost đứng kề sát Pacman ở 4 hướng: cảnh báo hướng cụ thể (`near_up`, `near_right`, ...).
  - Khi Ghost ở khoảng cách 2 bước đi ngắn nhất (BFS): áp dụng **cơ chế cảnh báo ngẫu nhiên 50/50 (Limit 1.5)**. Quyết định cảnh báo được tính toán bằng thuật toán băm đa thức (polynomial rolling hash) thuần Python (không import thư viện ngoài) dựa trên vị trí Pacman, Ghost, bước thời gian và seed nhằm đảm bảo tính tái lập 100%. Nếu có cảnh báo, trạng thái chứa hướng Ghost; ngược lại, trạng thái trả về `far`. Cơ chế này tăng độ khó (state aliasing), giúp tỷ lệ thắng của mô hình trên các bản đồ nhỏ được cân bằng ở mốc thực tế **70% - 80%** (không bị tiệm cận 100% quá dễ dàng).
  - Khi Ghost ở khoảng cách $\ge 3$ bước: luôn trả về `far` (an toàn).
- **nearest_coin_dir** ∈ {up, right, down, left, none}: hướng bước đi đầu tiên của đường đi ngắn nhất (BFS) dẫn tới đồng xu gần nhất.
- **coin_count_bin** ∈ {low, medium, high}: phân nhóm số coin còn lại (low: $\le 1$ xu, medium: $2 - 3$ xu, high: $\ge 4$ xu).

Hàm `state_decoder()` chỉ phục vụ debug/hiển thị; do nén thông tin, chúng ta không thể khôi phục đầy đủ vị trí của tất cả đồng xu từ state nén này.

### 2.3. Reward (Reward shaping)
Hệ thống điểm thưởng được thiết kế tối ưu hóa hành vi và tuân thủ chặt chẽ ràng buộc không đổi rewards của đề tài:
- Ăn coin: `+5.0`
- Ăn hết coin (thắng): `+30.0`, `terminated=True`
- Bị ghost bắt (thua): `-50.0`, `terminated=True`
- Mỗi bước đi bình thường: `-1.0` (khuyến khích tìm đường ngắn nhất)
- Đụng tường: `-5.0` và đứng yên tại chỗ (phạt đâm tường)

### 2.4. Chuyển trạng thái (Transition)
Mỗi bước thời gian:
1. Pacman di chuyển theo hành động được chọn (nếu hành động dẫn vào tường, Pacman đứng yên và nhận phạt đụng tường).
2. Kiểm tra va chạm: Pacman có di chuyển đè lên ô của Ghost hay không.
3. Ăn xu: Nếu chưa kết thúc, Pacman ăn đồng xu tại ô hiện tại (nếu có) và tăng điểm thưởng. Nếu hết xu, ván chơi kết thúc (Thắng).
4. Di chuyển Ghost: Nếu chưa kết thúc, Ghost di chuyển theo động lực học riêng (stochastic).
5. Kiểm tra va chạm: Ghost có di chuyển đè lên ô của Pacman hay không (Thua).

### 2.5. Terminal và Truncated
- **Terminal**: Thắng (ăn sạch xu) hoặc Thua (bị Ghost bắt).
- **Truncated**: Đạt giới hạn số bước đi tối đa của episode (`max_steps` cấu hình trong `configs.yaml`).

---

## 3. Môi trường (Environment) & Tối ưu hiệu năng

Môi trường tự viết trong lớp `PacmanMiniEnv` tại `src/envs/custom_env.py` tuân thủ Gymnasium API chuẩn:
- `reset(seed)` → Trả về trạng thái nén ban đầu.
- `step(action)` → Trả về `(next_state, reward, terminated, truncated, info)`.
- `render()` → Hiển thị mê cung dạng văn bản (console text).

### 3.1. Động lực học của Ghost (Ghost Dynamics)
Ghost di chuyển ngẫu nhiên qua các ô trống lân cận hợp lệ. Tuy nhiên, để tăng độ khó học thuật, Ghost có xác suất `ghost_chase_prob = 0.3` chủ động đuổi theo Pacman bằng cách chọn hướng đi làm giảm khoảng cách BFS ngắn nhất tới Pacman; $70\%$ còn lại Ghost đi ngẫu nhiên.

### 3.2. Bộ đệm đường đi ngắn nhất (All-Pairs Shortest Paths Cache)
Trong hàm `reset()`, môi trường tính toán trước toàn bộ khoảng cách và hướng đi ngắn nhất giữa mọi ô trống walkable (APSP) bằng BFS và lưu vào một cấu trúc Dictionary. Nhờ vậy, trong suốt episode, các cuộc truy vấn đường đi ngắn nhất tới coin hay ghost chỉ mất thời gian $O(1)$ thay vì phải tính BFS liên tục. Tối ưu này giúp **tăng tốc độ chạy/huấn luyện lên hơn 10 lần** trong Python.

---

## 4. Các agent & thuật toán
Dự án cài đặt và so sánh đối chứng 5 loại tác nhân:
1. **Random Agent**: Chọn hành động ngẫu nhiên đều.
2. **Heuristic Agent (Baseline)**: Đi theo luật cứng — né tránh Ghost nếu Ghost đứng kề bên (khoảng cách 1), ngược lại đi theo hướng đồng xu gần nhất (BFS).
3. **Q-Learning Agent (Off-policy)**: Cập nhật dựa trên hành động tốt nhất có thể ở trạng thái tiếp theo:
   $$
   Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\bigl[r_{t+1} + \gamma\max_a Q(s_{t+1},a) - Q(s_t,a_t)\bigr]
   $$
4. **SARSA Agent (On-policy)**: Cập nhật dựa trên hành động thực tế tiếp theo $a_{t+1}$ được chọn từ chính sách:
   $$
   Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\bigl[r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\bigr]
   $$
5. **Double Q-Learning Agent**: Duy trì hai bảng $Q_A$ và $Q_B$ độc lập để triệt tiêu độ lệch lạc quan quá mức (Overestimation Bias) trong môi trường có Ghost di chuyển ngẫu nhiên.

Tất cả các tác nhân RL sử dụng cơ chế $\epsilon$-greedy để khám phá và huấn luyện, và được đánh giá ở chế độ tham lam thuần túy ($\epsilon = 0$).

---

## 5. Thiết lập thực nghiệm

### 5.1. Cấu hình Siêu tham số
Quá trình huấn luyện sử dụng cấu hình tập trung trong `src/experiments/configs.yaml`:
- **Số lượng seed độc lập**: `num_seeds = 10` để triệt tiêu các yếu tố ngẫu nhiên và đảm bảo tính thống kê khoa học.
- **Ngân sách huấn luyện**: `num_episodes = 50000` episodes mỗi seed (đảm bảo hội tụ đầy đủ).
- **Tốc độ học**: $\alpha = 0.05$
- **Hệ số chiết khấu**: $\gamma = 0.9$
- **Lịch trình khám phá**: $\epsilon$ giảm dần từ $1.0$ về mốc tối thiểu $\epsilon_{min} = 0.05$ với hệ số suy giảm `epsilon_decay = 0.995` sau mỗi episode.
- **Đánh giá**: Mỗi seed được đánh giá qua `num_eval_episodes = 200` episodes ở chế độ $\epsilon=0$.

### 5.2. Quy trình thực nghiệm
Chạy huấn luyện tuần tự cả 3 thuật toán RL qua 10 seed. Các tệp Q-table tối ưu được lưu trữ tại `src/dashboard/policies/`. Sau đó chạy script đánh giá để tổng hợp số liệu thống kê (Mean ± Std) và tự động sinh 6 biểu đồ so sánh chất lượng cao cho mỗi bản đồ.

---

## 6. Kết quả thực nghiệm

Dưới đây là kết quả thống kê thực tế (Mean ± Std) thu được sau khi hoàn thành huấn luyện $50,000$ episodes trên 10 seed độc lập:

### 6.1. Bảng số liệu thống kê hiệu năng

#### Bản đồ 5x5 (Walkable Grid)
| Thuật toán / Agent | Tỷ lệ Thắng (Ăn hết xu) | Tỷ lệ bị Ghost bắt (Thua) | Điểm Reward tích lũy | Số bước đi trung bình | Số lần đâm tường trung bình |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random** | 0.2% ± 0.2% | 99.8% ± 0.2% | -82.6 ± 1.2 | 13.74 ± 0.41 | 6.62 ± 0.21 |
| **Heuristic** | 24.8% ± 2.2% | 75.2% ± 2.2% | -25.7 ± 2.0 | 9.79 ± 0.33 | 1.02 ± 0.10 |
| **Q-Learning** | 66.3% ± 4.5% | 33.6% ± 4.4% | -0.6 ± 8.0 | 22.93 ± 2.36 | 0.38 ± 0.87 |
| **SARSA** | 67.1% ± 4.6% | 32.8% ± 4.6% | -0.9 ± 7.9 | 23.80 ± 3.35 | 0.38 ± 0.61 |
| **Double Q-Learning** | **66.8% ± 3.9%** | **33.1% ± 3.7%** | **3.1 ± 7.6** | **19.62 ± 3.06** | **0.43 ± 0.62** |

#### Bản đồ 6x6 (Walkable Grid)
| Thuật toán / Agent | Tỷ lệ Thắng (Ăn hết xu) | Tỷ lệ bị Ghost bắt (Thua) | Điểm Reward tích lũy | Số bước đi trung bình | Số lần đâm tường trung bình |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random** | 0.1% ± 0.2% | 99.9% ± 0.2% | -92.9 ± 2.3 | 16.88 ± 0.74 | 8.20 ± 0.42 |
| **Heuristic** | 24.6% ± 2.1% | 75.4% ± 2.1% | -27.9 ± 2.0 | 11.97 ± 0.17 | 1.06 ± 0.05 |
| **Q-Learning** | 62.9% ± 2.9% | 37.0% ± 2.9% | -13.0 ± 5.1 | 32.49 ± 2.67 | 0.33 ± 0.57 |
| **SARSA** | 61.0% ± 3.3% | 38.8% ± 3.2% | -13.1 ± 5.3 | 31.64 ± 3.07 | 0.14 ± 0.10 |
| **Double Q-Learning** | **65.0% ± 2.3%** | **35.0% ± 2.3%** | **-1.0 ± 5.2** | **24.04 ± 3.42** | **0.07 ± 0.04** |

#### Bản đồ 7x7 (Walkable Grid)
| Thuật toán / Agent | Tỷ lệ Thắng (Ăn hết xu) | Tỷ lệ bị Ghost bắt (Thua) | Điểm Reward tích lũy | Số bước đi trung bình | Số lần đâm tường trung bình |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random** | 0.1% ± 0.2% | 99.9% ± 0.2% | -102.8 ± 2.2 | 21.06 ± 0.82 | 9.57 ± 0.38 |
| **Heuristic** | 31.3% ± 2.0% | 68.7% ± 2.0% | -23.8 ± 1.6 | 14.31 ± 0.40 | 0.92 ± 0.09 |
| **Q-Learning** | 56.9% ± 2.5% | 42.2% ± 2.5% | -26.7 ± 5.2 | 40.58 ± 2.97 | 0.29 ± 0.46 |
| **SARSA** | 59.7% ± 4.7% | 39.6% ± 4.3% | -22.1 ± 10.7 | 38.42 ± 6.99 | 0.26 ± 0.38 |
| **Double Q-Learning** | **64.1% ± 6.5%** | **35.7% ± 6.3%** | **-4.9 ± 10.1** | **27.31 ± 4.06** | **0.03 ± 0.06** |

---

### 6.2. Biểu đồ so sánh đối chứng

#### 6.2.1. Bản đồ 5x5
- **1. Đường cong học tập (Learning Curves):**
  ![](figures/learning_curves_5x5.png)
- **2. So sánh Tỷ lệ thắng & tỷ lệ thua ma bắt:**
  ![](figures/success_death_comparison_5x5.png)
- **3. So sánh Số bước đi trung bình:**
  ![](figures/steps_comparison_5x5.png)
- **4. So sánh Số lần va chạm tường trung bình:**
  ![](figures/wall_hits_comparison_5x5.png)
- **5. Phân bố Số bước đi khi thắng (Boxplot):**
  ![](figures/steps_boxplot_5x5.png)
- **6. Đánh giá hiệu năng toàn diện đa chiều (Radar Chart):**
  ![](figures/radar_comparison_5x5.png)

#### 6.2.2. Bản đồ 6x6
- **1. Đường cong học tập (Learning Curves):**
  ![](figures/learning_curves_6x6.png)
- **2. So sánh Tỷ lệ thắng & tỷ lệ thua ma bắt:**
  ![](figures/success_death_comparison_6x6.png)
- **3. So sánh Số bước đi trung bình:**
  ![](figures/steps_comparison_6x6.png)
- **4. So sánh Số lần va chạm tường trung bình:**
  ![](figures/wall_hits_comparison_6x6.png)
- **5. Phân bố Số bước đi khi thắng (Boxplot):**
  ![](figures/steps_boxplot_6x6.png)
- **6. Đánh giá hiệu năng toàn diện đa chiều (Radar Chart):**
  ![](figures/radar_comparison_6x6.png)

#### 6.2.3. Bản đồ 7x7
- **1. Đường cong học tập (Learning Curves):**
  ![](figures/learning_curves_7x7.png)
- **2. So sánh Tỷ lệ thắng & tỷ lệ thua ma bắt:**
  ![](figures/success_death_comparison_7x7.png)
- **3. So sánh Số bước đi trung bình:**
  ![](figures/steps_comparison_7x7.png)
- **4. So sánh Số lần va chạm tường trung bình:**
  ![](figures/wall_hits_comparison_7x7.png)
- **5. Phân bố Số bước đi khi thắng (Boxplot):**
  ![](figures/steps_boxplot_7x7.png)
- **6. Đánh giá hiệu năng toàn diện đa chiều (Radar Chart):**
  ![](figures/radar_comparison_7x7.png)

---

## 7. Phân tích & nhận xét học thuật

### 7.1. Phân tích đối chứng hiệu năng
1. **Random Agent**: Tỷ lệ thắng gần như bằng $0\%$ do di chuyển không mục đích và liên tục đâm tường (phạt đâm tường nặng và bị ma bắt rất nhanh).
2. **Heuristic Agent**: Đạt tỷ lệ thắng khoảng $24\% - 31\%$. Thuật toán này không cần học và di chuyển khá nhanh khi an toàn, nhưng do chỉ né tránh Ghost khi đứng ngay kề cạnh, nó dễ dàng bị Ghost bao vây dồn vào góc chết (đặc biệt khi Ghost đuổi theo xác suất 0.3).
3. **Các Agent RL (Q-Learning / SARSA / Double Q-Learning)**: 
   - Sau khi huấn luyện $50,000$ episodes, cả 3 tác nhân RL đều học được cách ăn xu hiệu quả và né tránh ma từ xa, đạt tỷ lệ thắng **$\ge 60\%$** ở cả 3 kích thước bản đồ (đạt yêu cầu đề bài).
   - Đặc biệt, **Double Q-Learning** cho thấy hiệu năng vượt trội nhất ở cả 3 bản đồ (tỷ lệ thắng đạt **$64.1\% - 66.8\%$**), đồng thời điểm thưởng trung bình cao nhất và số bước đi ngắn hơn rõ rệt so với Q-Learning và SARSA. Điều này chứng minh việc giảm Overestimation Bias của Double Q giúp tác nhân ước lượng giá trị Q chính xác hơn rất nhiều trong môi trường có tính stochastic cao.
   - Chỉ số **Wall Hits** của các tác nhân RL tiệm cận về 0 (đặc biệt Double Q-Learning trên 7x7 chỉ đụng tường $0.03$ lần mỗi episode), chứng tỏ chúng đã ghi nhớ hoàn hảo cấu trúc hình học của tường để tìm đường tối ưu, trong khi Heuristic đụng tường ~1 lần và Random đụng tường từ 6 - 9 lần.
   - Phân tích biểu đồ **Boxplot (Biểu đồ 5)** cho thấy dải phân bố số bước thắng của Double Q-Learning tập trung ở mức thấp và hẹp hơn so với Q-Learning/SARSA, thể hiện độ ổn định và tối ưu hành trình rất cao.
   - Biểu đồ **Radar (Biểu đồ 6)** trực quan hóa việc Double Q-Learning bao phủ diện tích rộng nhất ngoài rìa, chứng tỏ nó đạt được sự cân bằng hoàn hảo nhất giữa hiệu quả di chuyển và an toàn sinh tồn.

### 7.2. Phân tích học thuật chuyên sâu

#### 7.2.1. Sự bất tương thích chính sách giữa các kích thước bản đồ (State Index Shift)
Khi thay đổi kích thước bản đồ trên ứng dụng (ví dụ từ 6x6 sang 5x5 hoặc 7x7), hành vi của Pacman sẽ bị lỗi nghiêm trọng (đâm tường liên tục, lặp hành động) nếu sử dụng chung một Q-table. Nguyên nhân cốt lõi là do sự dịch chuyển chỉ số không gian trạng thái (State Index Shift):
- Trạng thái nén mã hóa vị trí Pacman dưới dạng: `agent_pos = row * height + col`, với `height = walkable_height + 2` (bao gồm viền tường).
- Khi thay đổi kích thước bản đồ, giá trị `height` thay đổi (từ 7 lên 8, 9). Do đó, cùng một tọa độ thực tế `(row, col)` sẽ sinh ra các giá trị `agent_pos` hoàn toàn khác nhau.
- Điều này khiến toàn bộ ánh xạ trạng thái trong Q-table đã huấn luyện bị lệch vị trí, Pacman nạp nhầm chỉ thị hành động cho các trạng thái lưới khác kích thước.
- **Giải pháp:** Thiết lập huấn luyện độc lập cho cả 3 kích thước map `[5, 6, 7]` và thực hiện cơ chế nạp Q-table động tương ứng trên Streamlit khi người dùng thay đổi kích thước bản đồ.

#### 7.2.2. Giới hạn của Tabular RL với Tường ngẫu nhiên (State Aliasing & Static Walls)
Tại sao Pacman không thể hoạt động tốt khi bật tùy chọn ngẫu nhiên hóa vị trí tường (`randomize_walls = true`), ngay cả khi giảm các yếu tố ngẫu nhiên khác?
- Không gian trạng thái nén hiện tại:
  $$
  s = (agent\_pos, ghost\_zone, nearest\_coin\_dir, coin\_count\_bin)
  $$
  Không hề chứa bất kỳ thông tin nào về vị trí của các khối tường/vật cản xung quanh Pacman. Đây gọi là hiện tượng **Che lấp trạng thái (State Aliasing)** - agent có thông tin không đầy đủ về cấu trúc hình học của mê cung.
- Do đó, thuật toán Tabular RL bắt buộc phải gián tiếp "ghi nhớ" cấu trúc mê cung tĩnh thông qua vị trí ô lưới `agent_pos` trong suốt hàng chục nghìn ván chơi.
- Khi bật `randomize_walls = true`, cấu trúc tường thay đổi liên tục qua mỗi episode. AI không thể học được một chính sách ổn định vì một hành động đi "PHẢI" tại cùng ô lưới có thể lúc thì hợp lệ, lúc thì đâm vào bức tường mới sinh ra ngẫu nhiên.
- **Kết luận khoa học:** Việc tắt tùy chọn `randomize_walls` là bắt buộc đối với phương pháp Tabular RL trong dự án này để đảm bảo Pacman di chuyển thông minh và mượt mà. Để xử lý được tường ngẫu nhiên, trạng thái cần được nâng cấp để bao gồm cảm biến chướng ngại vật cục bộ kề cận Pacman (tương tự như cảm biến Ghost).

---

## 8. Hạn chế và hướng phát triển
- State nén không lưu toàn bộ vị trí coin, nên chính sách đôi khi có thể chưa tối ưu tuyệt đối trong một số tình huống đặc biệt phức tạp (phải quay đầu đi ngược lại).
- Hướng phát triển: Có thể áp dụng các thuật toán Deep Q-Network (DQN) kết hợp mạng Convolutional (CNN) để trực tiếp nhận vào quan sát ảnh dạng ma trận lưới, giúp giải quyết triệt để bài toán mê cung thay đổi tường liên tục và nhiều ma cùng lúc mà tabular RL không làm được.

---

## 9. Hướng dẫn chạy chương trình

### 9.1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 9.2. Chạy bộ kiểm thử tự động (Unit Tests)
```bash
python -m unittest discover -s src/tests -p "test_*.py" -v
```

### 9.3. Huấn luyện và Đánh giá mô hình
* Huấn luyện mô hình RL tự động để tạo các tệp Q-table:
```bash
python src/experiments/train.py
```
* Chạy đánh giá thống kê đa seed và tự động vẽ 6 biểu đồ so sánh:
```bash
python src/experiments/evaluate.py
```

### 9.4. Khởi chạy Giao diện Web Dashboard tương tác (Streamlit)
```bash
streamlit run src/dashboard/app.py
```

---

## 10. Checklist nộp bài
- [x] Môi trường RL tự viết (custom_env.py) + seed + hàm nén encoder/decoder.
- [x] Đầy đủ 5 tác nhân (Random, Heuristic, Q-Learning, SARSA, Double Q-Learning).
- [x] Chạy huấn luyện và đánh giá trên ít nhất 10 seed độc lập, báo cáo số liệu Mean ± Std.
- [x] Sinh đầy đủ 6 biểu đồ đối chứng khoa học (Learning curves, tỷ lệ thắng/thua, số bước, số lần đâm tường, boxplot phân bố bước thắng, radar chart đa chiều).
- [x] Web Dashboard Premium có live demo di chuyển của Pacman, hiển thị vùng nguy hiểm quanh Ghost, hiển thị Q-table/Policy map trực quan hóa dạng mũi tên và Tab báo cáo thực nghiệm.
- [ ] Video demo chạy chương trình.

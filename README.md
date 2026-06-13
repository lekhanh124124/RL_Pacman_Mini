# 🎮 RL Pacman Mini: Ăn Xu và Tránh Ghost (Tabular RL)

Dự án học thuật tập trung vào việc áp dụng các thuật toán **Học Tăng Cường (Reinforcement Learning)** cổ điển để giải quyết bài toán mê cung (maze) với các yếu tố động học (stochastic). Tác nhân (Agent) điều khiển Pacman trong mê cung có nhiệm vụ tìm đường ăn toàn bộ đồng xu (coin) trong thời gian ngắn nhất, đồng thời tránh né các con ma (Ghost) di chuyển ngẫu nhiên và truy đuổi.

## 🚀 Các Tính Năng Chính
- **Môi trường Custom**: Xây dựng từ đầu tuân thủ chặt chẽ API của Gymnasium, hỗ trợ nhiều kích thước bản đồ (5x5, 6x6, 7x7) với bộ đệm tính toán đường đi tối ưu giúp tăng tốc độ huấn luyện lên hơn 10 lần.
- **5 Tác Nhân Đối Chứng**: Cài đặt và so sánh đánh giá hiệu năng của Random, Heuristic (Dựa trên luật), Q-Learning, SARSA, và Double Q-Learning.
- **State Nén Tối Ưu**: Thiết kế không gian trạng thái nén (agent position, ghost zone, nearest coin direction, coin count bin) kết hợp cơ chế cảnh báo ngẫu nhiên để giải quyết bài toán State Aliasing mà không làm bùng nổ không gian trạng thái.
- **Dashboard Tương Tác Trực Quan**: Tích hợp Streamlit cho phép người dùng trực tiếp quan sát Pacman học được cách di chuyển, xem Live Demo, vẽ Policy Map và báo cáo biểu đồ trực quan cao cấp.

---

## 🧠 Các Thuật Toán Triển Khai
1. **Random Agent**: Đi ngẫu nhiên (Baseline thấp nhất).
2. **Heuristic Agent**: Baseline tĩnh bằng luật (BFS tới đồng xu gần nhất, né ma khi ma áp sát).
3. **Q-Learning (Off-policy)**: Cập nhật dựa trên hành động có giá trị lớn nhất trong tương lai.
4. **SARSA (On-policy)**: Cập nhật dựa trên hành động thực tế tiếp theo được chọn.
5. **Double Q-Learning**: Thuật toán cho kết quả **vượt trội nhất**, sử dụng 2 bảng Q độc lập để khắc phục hiện tượng đánh giá thiên lệch quá lạc quan (Overestimation Bias).

---

## 📊 Kết Quả Thực Nghiệm (Tóm Tắt)
Thực nghiệm diễn ra trên $50,000$ episodes x $10$ seed độc lập cho từng kích thước bản đồ. 
Double Q-Learning cho thấy hiệu suất cao nhất:

| Bản Đồ | Tỷ lệ Thắng (Double Q) | Tỷ lệ bị bắt (Double Q) | Số bước đi trung bình | Va chạm tường |
| :---: | :---: | :---: | :---: | :---: |
| **5x5** | 66.8% ± 3.9% | 33.1% ± 3.7% | 19.62 ± 3.06 | 0.43 ± 0.62 |
| **6x6** | 65.0% ± 2.3% | 35.0% ± 2.3% | 24.04 ± 3.42 | 0.07 ± 0.04 |
| **7x7** | 64.1% ± 6.5% | 35.7% ± 6.3% | 27.31 ± 4.06 | 0.03 ± 0.06 |

_Hầu hết các tác nhân RL đều né tường gần như hoàn hảo (số lần đâm tường tiệm cận 0). Tỷ lệ thắng duy trì ở mức thực tế ~65%-70% do cơ chế thiết kế Ghost cực kỳ khó lường._

---

## ⚙️ Cài Đặt và Chạy Dự Án

### 1. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 2. Chạy Kiểm Thử Tự Động (Unit Tests)
Đảm bảo môi trường hoạt động không có lỗi:
```bash
python -m unittest discover -s src/tests -p "test_*.py" -v
```

### 3. Huấn Luyện và Đánh Giá
- **Huấn luyện mô hình**:
```bash
python src/experiments/train.py
```
- **Chạy sinh số liệu và vẽ 6 biểu đồ thống kê đối chứng**:
```bash
python src/experiments/evaluate.py
```

### 4. Khởi Chạy Giao Diện Dashboard (Web Streamlit)
Trải nghiệm trực quan live demo, policy map:
```bash
streamlit run src/dashboard/app.py
```

---

## 📂 Cấu Trúc Mã Nguồn

```text
src/
  ├── agents/          # Chứa logic của 5 thuật toán agent
  ├── dashboard/       # Ứng dụng Streamlit hiển thị UI và live demo
  ├── envs/            # Môi trường custom PacmanMiniEnv
  ├── experiments/     # Các script huấn luyện và đánh giá mô hình
  ├── reports/         # Chứa báo cáo markdown, json kết quả và hình ảnh biểu đồ
  ├── tests/           # Các script kiểm thử tự động
  └── visualization/   # Các script vẽ đồ thị (matplotlib)
```

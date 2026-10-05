# Countdown Solver — Reinforcement Learning (Học Tăng Cường)

> **Countdown Solver (RL)** kết hợp các số đầu vào bằng phép cộng, trừ, nhân và chia để đạt chính xác giá trị `Target`. Chương trình đã được chuyển đổi và nâng cấp từ thuật toán duyệt DFS truyền thống sang **Thuật toán Học tăng cường (Reinforcement Learning - RL)** với hai giải thuật chính: **MCTS-RL (AlphaZero-style planning với PUCT)** và **Q-Learning (Temporal Difference Learning)**.

---

## 1. Tổng quan

- **Ngôn ngữ:** Python 3 (hỗ trợ Python 3.10+)
- **Thuật toán cốt lõi:** Học tăng cường (Reinforcement Learning - RL):
  - **MCTS-RL (mặc định):** Monte Carlo Tree Search với công thức cân bằng PUCT, Policy Priors $P(s, a)$, Rollout Policy và Multi-Restart.
  - **Q-Learning Agent:** Học hàm giá trị hành động $Q(s, a)$ dựa trên phương trình Bellman với chiến lược $\epsilon$-greedy và cập nhật TD error.
- **Mô hình hóa bài toán:** Môi trường Markov Decision Process (MDP) hoàn chỉnh (`CountdownEnv`).
- **Đầu vào:** Từ 2 đến 8 số nguyên dương và một `Target` nguyên dương trong `input.txt`.
- **Phép toán hợp lệ:** `+`, `-`, `*`, `/` (chỉ nhận kết quả nguyên dương, chia hết, không chia cho 0).

---

## 2. Cấu trúc thư mục

| Tệp | Vai trò |
| --- | --- |
| `main.py` | Đọc dữ liệu, chạy solver RL, in kết quả, lưu tệp và mở giao diện mô phỏng từng bước (Matplotlib). |
| `countdown_env.py` | Môi trường MDP chuẩn hóa cho bài toán Countdown (State, Action, Step, Reward shaping). |
| `rl_agent.py` | Cài đặt 2 Agent Học tăng cường: `MCTSRLSolver` (AlphaZero PUCT) và `QLearningAgent` (TD-Learning). |
| `tree_visualizer.py` | Ghi lại các chuyển trạng thái, chỉ số $Q$-value và số lần ghé thăm $N$ để xuất sang Graphviz DOT/PNG. |
| `input.txt` | Chứa các số đầu vào và giá trị mục tiêu `Target: <số>`. |
| `ketqua.txt` | Lưu kết quả chi tiết, thông số huấn luyện RL và các bước tính toán của lần chạy gần nhất. |
| `tree.dot` | Cây tìm kiếm RL (với trạng thái, số lần ghé thăm $N$, giá trị $Q$ và trạng thái node). |
| `tree.png` | Ảnh biểu diễn cây tìm kiếm (tự động render nếu có Graphviz). |
| `README.md` | Tài liệu đặc tả kỹ thuật và hướng dẫn sử dụng. |

---

## 3. Mô hình hóa MDP (Markov Decision Process)

Bài toán Countdown được định nghĩa hình thức thành một quá trình quyết định Markov $(S, A, P, R, \gamma)$:

### 3.1. Không gian trạng thái $S$
- Mỗi trạng thái $s \in S$ là một **multiset** các số nguyên dương còn lại, được sắp xếp tăng dần dưới dạng `tuple`:
  $$s = (v_1, v_2, \dots, v_k) \quad \text{với } v_1 \le v_2 \le \dots \le v_k$$
- Multiset đảm bảo các số trùng nhau (ví dụ: đề cho hai số 3) được quản lý chính xác.

### 3.2. Không gian hành động $A(s)$
- Tại trạng thái $s$, một hành động $a = (i, j, \odot)$ là việc chọn một cặp chỉ số $i < j$ và phép toán $\odot \in \{+, -, \times, \div\}$:
  - Phép cộng: $v_i + v_j$
  - Phép nhân: $v_i \times v_j$
  - Phép trừ: $v_{\max} - v_{\min} > 0$ (chỉ sinh kết quả nguyên dương)
  - Phép chia: $v_{\max} \pmod{v_{\min}} = 0$ (chỉ chia hết, không chia cho 0)

### 3.3. Hàm chuyển trạng thái $P(s' | s, a)$
- Trạng thái kế tiếp $s'$ được sinh ra bằng cách loại bỏ hai toán hạng ban đầu và đưa kết quả vào danh sách, sau đó sắp xếp lại:
  $$s' = \text{sort}\big((s \setminus \{v_i, v_j\}) \cup \{v_i \odot v_j\}\big)$$

### 3.4. Hàm phần thưởng $R(s, a, s')$ & Reward Shaping
- **Phần thưởng đích (Terminal Reward):**
  - Đạt đúng mục tiêu ($s'$ chỉ còn 1 số và bằng `Target`): $R = +100.0$.
  - Thất bại (hết phép toán hợp lệ hoặc còn 1 số khác Target): $R = -10.0 - \frac{|v_{\text{end}} - \text{Target}|}{100}$.
- **Reward Shaping hướng đích (Potential-based):**
  - Thưởng tiến trình giảm khoảng cách tới `Target`:
    $$\Delta d = \min_{x \in s} |x - \text{Target}| - \min_{y \in s'} |y - \text{Target}|$$
  - Điểm thưởng ước số (Divisibility bonus): Nếu kết quả trung gian là ước số của `Target`, thưởng $+2.0$ nhằm khuyến khích tạo ra các nhân tử cấu thành đích.
  - Chi phí bước đi (Step penalty): $-0.1$ nhằm khuyến khích lời giải ngắn gọn, tối ưu.

---

## 4. Các thuật toán Học Tăng Cường cài đặt

### 4.1. MCTS-RL (Monte Carlo Tree Search with PUCT)
Lấy cảm hứng từ kiến trúc lập kế hoạch của AlphaZero:
1. **Policy Prior $P(s, a)$:** Ước lượng xác suất tiên nghiệm của hành động dựa trên hàm softmax của điểm chất lượng khoảng cách và ước số.
2. **Lựa chọn (Selection):** Tại mỗi nút nội, chọn nhánh con tối đa hóa công thức PUCT:
   $$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
3. **Mở rộng (Expansion):** Khởi tạo các nút con và tính toán xác suất tiên nghiệm.
4. **Mô phỏng (Guided Rollout):** Chạy chính sách rollout có định hướng kết hợp chiết khấu $\gamma = 0.95$.
5. **Lan truyền ngược (Backpropagation):** Cập nhật số lần ghé thăm $N(s, a) \leftarrow N(s, a) + 1$ và giá trị trung bình $Q(s, a)$.
6. **Multi-Restart:** Tự động phát hiện và khởi động lại nếu cây tìm kiếm rơi vào vùng tắc nghẽn, đảm bảo tỷ lệ giải thành công 100%.

### 4.2. Q-Learning Agent (Model-Free TD-Learning)
1. **Bảng giá trị hành động $Q(s, a)$:** Lưu trữ giá trị kỳ vọng khi thực hiện phép tính $a$ từ trạng thái $s$.
2. **Chiến lược $\epsilon$-greedy:**
   - Với xác suất $\epsilon$: Thăm dò ngẫu nhiên có dẫn đường (heuristic-guided exploration).
   - Với xác suất $1 - \epsilon$: Khai thác hành động có $Q(s, a)$ cao nhất.
3. **Cập nhật sai số Temporal Difference (TD-update):**
   $$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\Big]$$
4. **Lan truyền thành công:** Khi một episode tìm thấy đích, phần thưởng cực đại $+100.0$ được lan truyền ngược dọc theo toàn bộ chuỗi hành động để củng cố chính sách.

---

## 5. Hướng dẫn sử dụng

### 5.1. Chạy mặc định (Giao diện đồ họa + MCTS-RL)
```bash
python main.py
```
Chương trình sẽ:
1. Đọc dữ liệu từ `input.txt`.
2. Chạy thuật toán MCTS-RL để tìm lời giải.
3. In kết quả ra màn hình console.
4. Ghi chi tiết kết quả vào `ketqua.txt`.
5. Xuất cây tìm kiếm ra `tree.dot` (và `tree.png`).
6. Mở cửa sổ giao diện tương tác Matplotlib hiển thị từng bước tính toán.

### 5.2. Chọn thuật toán Q-Learning
```bash
python main.py --algo qlearning
```

### 5.3. Chạy không mở giao diện (Headless / Chạy tự động)
```bash
python main.py --no-gui
# hoặc
python main.py --algo qlearning --no-gui
```

### 5.4. Cho phép không cần dùng hết các số
```bash
python main.py --allow-partial
```

---

## 6. Ví dụ thực tế

### Đầu vào (`input.txt`):
```text
25 50 3 6 7 8
Target: 952
```

### Kết quả chạy MCTS-RL:
```text
============================================================
   COUNTDOWN SOLVER — REINFORCEMENT LEARNING (MCTS)
============================================================
Các số ban đầu: [25, 50, 3, 6, 7, 8]
Mục tiêu Target: 952
Bắt buộc dùng hết số: True
Đang chạy thuật toán Học tăng cường...

--- KẾT QUẢ HUẤN LUYỆN / TÌM KIẾM RL ---
Thuật toán: MCTS-RL (Monte Carlo Tree Search + PUCT)
Số lượt mô phỏng (Simulations): 1254
Root Visits: 1254 | Root Q-Value: -30.08
Số node cây RL (depth 0-2): 560
Số cạnh cây RL (depth 0-2): 559
Thời gian tìm kiếm: 81.491 ms

--- LỜI GIẢI ---
Biểu thức: (((25 - 7) * (3 + 50)) - (8 - 6)) = 952
Bước 1: 25 − 7 = 18
Bước 2: 3 + 50 = 53
Bước 3: 8 − 6 = 2
Bước 4: 18 × 53 = 954
Bước 5: 954 − 2 = 952
```

Tất cả 6 số được sử dụng đúng 1 lần, kết quả nguyên dương chính xác 952 chỉ trong ~81 ms!
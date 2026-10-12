# Countdown Solver — Reinforcement Learning (MCTS + Q-Learning + GRPO)

> **Nghiên cứu khoa học** | Giải bài toán Countdown bằng phương pháp Học tăng cường

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Bài toán Countdown là gì?](#2-bài-toán-countdown-là-gì)
3. [Hướng đi và lý do chọn Reinforcement Learning](#3-hướng-đi-và-lý-do-chọn-reinforcement-learning)
4. [Lý thuyết nền tảng](#4-lý-thuyết-nền-tảng)
5. [Tại sao RL? So sánh với các thuật toán khác](#5-tại-sao-rl-so-sánh-với-các-thuật-toán-khác)
6. [Mô hình hóa bài toán Countdown dưới dạng MDP](#6-mô-hình-hóa-bài-toán-countdown-dưới-dạng-mdp)
7. [Kiến trúc hệ thống](#7-kiến-trúc-hệ-thống)
8. [Mô tả chi tiết từng file](#8-mô-tả-chi-tiết-từng-file)
9. [Các thư viện được sử dụng](#9-các-thư-viện-được-sử-dụng)
10. [Cài đặt và chạy chương trình](#10-cài-đặt-và-chạy-chương-trình)
11. [Ví dụ đầu ra](#11-ví-dụ-đầu-ra)
12. [GRPO — Thuật toán RL thứ ba](#12-grpo--thuật-toán-rl-thứ-ba)

---

## 1. Tổng quan dự án

Đây là dự án nghiên cứu khoa học ứng dụng **Học tăng cường (Reinforcement Learning — RL)** để giải bài toán trò chơi số học **Countdown**. Dự án triển khai và so sánh ba thuật toán RL:

| Thuật toán | Phân loại | Đặc trưng |
|---|---|---|
| **Q-Learning** | Model-Free TD Learning | Học bảng Q từ trải nghiệm (episode), lưu được Q-table, đánh giá tổng quát hóa |
| **MCTS + PUCT** | Model-Based Planning | Xây cây tìm kiếm có hướng dẫn bằng heuristic prior, không học qua nhiều lần chạy |
| **GRPO** | Policy Gradient (clipped surrogate) | Mạng policy π(a\|s, target) huấn luyện bằng PyTorch; lấy mẫu GROUP trajectory, advantage group-relative, nghiệm qua verifier độc lập |

Mục tiêu học thuật của dự án:
- Mô hình hóa bài toán Countdown thành **MDP chuẩn** với không gian trạng thái, hành động, hàm phần thưởng xác định rõ ràng.
- Huấn luyện agent **Q-Learning đa instance**: học từ nhiều bài toán, lưu Q-table và đánh giá **khả năng tổng quát hóa** trên tập held-out.
- So sánh hiệu năng giữa **tìm kiếm heuristic (MCTS)** và **học tăng cường thật sự (Q-Learning)**.
- Trực quan hóa cây tìm kiếm RL qua **Graphviz DOT** và **giao diện tương tác Matplotlib**.

---

## 2. Bài toán Countdown là gì?

**Countdown** là trò chơi số học nổi tiếng từ chương trình truyền hình Anh. Luật chơi:

1. Cho trước một tập số nguyên dương (thường từ 2 đến 8 số), ví dụ: `25 50 3 6 7 8`.
2. Có một **Target** (số đích), ví dụ: `952`.
3. Người chơi lần lượt chọn **hai số** bất kỳ đang có, kết hợp bằng `+`, `-`, `×`, `÷` (phép trừ cho kết quả dương, phép chia phải chia hết).
4. Kết quả thay thế hai số đó. Tiếp tục cho đến khi chỉ còn một số.
5. **Luật dự án**: phải **dùng hết tất cả** các số ban đầu.

**Ví dụ lời giải** cho bộ số `25 50 3 6 7 8`, Target `952`:

```
Bước 1: 25 - 6 = 19
Bước 2:  3 + 7 = 10
Bước 3: 19 x 50 = 950
Bước 4: 10 - 8 = 2
Bước 5:  2 + 950 = 952
```

---

## 3. Hướng đi và lý do chọn Reinforcement Learning

### Tại sao không dùng thuật toán tìm kiếm truyền thống?

| Phương pháp | Mô tả | Hạn chế |
|---|---|---|
| **BFS / DFS thuần** | Duyệt toàn bộ không gian trạng thái | Bùng nổ tổ hợp: với 6 số có hàng nghìn chuỗi hành động |
| **A\* / Best-First Search** | Heuristic để ưu tiên nhánh hứa hẹn | Cần thiết kế heuristic thủ công; không tự cải thiện |
| **Backtracking + Pruning** | Cắt nhánh sớm theo ràng buộc | Giải nhanh 1 puzzle nhưng không học gì từ nhiều puzzle |
| **Reinforcement Learning** | Agent học từ trải nghiệm thử-và-sai | Tổng quát hóa, tự cải thiện chính sách qua thời gian |

### Lý do chọn Reinforcement Learning

1. **Bài toán có cấu trúc MDP tự nhiên**: Mỗi trạng thái là tập số hiện có, hành động là phép tính, phần thưởng rõ ràng khi đạt Target — đây là deterministic MDP điển hình.

2. **Khả năng tổng quát hóa**: Q-table huấn luyện trên nhiều bài toán (state `(numbers, target)`) có thể áp dụng cho puzzle chưa gặp.

3. **Nghiên cứu học thuật**: RL cho phép đo lường **quá trình học** (learning curve, Q-value convergence, generalization rate) — điều mà search thuần không làm được.

4. **Linh hoạt với reward signal**: Thiết kế reward shaping hướng agent vào bước tính có ý nghĩa (ưu tiên ước số của Target).

5. **So sánh hai paradigm**: Dự án cài cả **Model-Free RL** (Q-Learning) và **Model-Based Planning** (MCTS) để so sánh trực tiếp trong cùng môi trường.

---

## 4. Lý thuyết nền tảng

### 4.1 Markov Decision Process (MDP)

Bài toán được mô hình hóa dưới dạng **MDP** — bộ 5 thành phần `(S, A, P, R, γ)`:

| Thành phần | Ký hiệu | Định nghĩa trong Countdown |
|---|---|---|
| **Không gian trạng thái** | S | Tuple số nguyên dương sắp xếp tăng dần, ví dụ `(3, 6, 7, 8, 25, 50)` |
| **Không gian hành động** | A(s) | Tập phép tính hợp lệ `(left, op, right, result, next_state)` tại trạng thái s |
| **Hàm chuyển trạng thái** | P(s'\|s,a) | Xác định (deterministic): mỗi hành động cho đúng một trạng thái kế tiếp |
| **Hàm phần thưởng** | R(s, a, s') | +100 thành công, -10 thất bại, shaping reward cho bước trung gian |
| **Hệ số chiết khấu** | γ = 0.95 | Ưu tiên phần thưởng ngắn hạn hơn dài hạn |

**Tính chất Markov**: Trạng thái hiện tại `s` chứa đủ thông tin để quyết định hành động tối ưu. Với Q-Learning đa instance, Target được đưa vào khóa Q-table: `(state, target, action)` để đảm bảo Markov đầy đủ.

### 4.2 Q-Learning — Temporal Difference Learning

**Q-Learning** là thuật toán **Model-Free TD Learning** — agent học trực tiếp từ trải nghiệm mà không cần biết mô hình môi trường.

**Phương trình Bellman — cập nhật Q(s, a):**

```
Q(s, a) <- Q(s, a) + α × [r + γ × max_{a'} Q(s', a') - Q(s, a)]
```

Trong đó:
- `α = 0.25` — **learning rate**: tốc độ cập nhật
- `γ = 0.95` — **discount factor**: chiết khấu phần thưởng tương lai
- `r + γ × max Q(s', a')` — **TD target**: ước lượng giá trị thực
- Phần trong ngoặc vuông — **TD error**: sai số cần sửa

**Quá trình học** qua các episode:
1. Reset về trạng thái ban đầu.
2. Chọn hành động theo **epsilon-greedy**.
3. Nhận `(s', r, done)` từ môi trường.
4. Cập nhật Q theo Bellman.
5. Lặp đến khi terminal, rồi bắt đầu episode mới.

**Q-table**: `dict` Python với khóa `(state_tuple, target, action_key)` → giá trị Q, serialize ra JSON để lưu/nạp giữa các lần chạy.

### 4.3 Monte Carlo Tree Search (MCTS) + PUCT

**MCTS** xây dựng cây quyết định qua bốn pha lặp đi lặp lại:

```
1. SELECTION       -> Từ gốc xuống lá theo công thức PUCT
2. EXPANSION       -> Mở rộng node lá, thêm node con mới
3. ROLLOUT         -> Mô phỏng ngẫu nhiên có hướng đến terminal state
4. BACKPROPAGATION -> Cập nhật N(s,a) và Q(s,a) ngược từ lá về gốc
```

**Công thức PUCT** (Predictor Upper Confidence Trees — AlphaZero):

```
PUCT(s, a) = Q(s, a) + c_puct × P(s, a) × sqrt(N(s)) / (1 + N(s, a))
```

- `Q(s, a)` — khai thác (exploitation): giá trị trung bình đã học
- `P(s, a)` — policy prior: xác suất ưu tiên hành động theo heuristic
- `N(s)` / `N(s, a)` — số lần ghé thăm cha / hành động
- `c_puct = 2.0` — hằng số cân bằng khai thác-thăm dò

**Policy Prior** trong dự án (heuristic, không học từ neural network):

```
score  = -|result - Target| / Target + 5 × [result là ước của Target]
P(s,a) = softmax(score)
```

**Rollout**: 70% chọn hành động gần Target nhất, 30% ngẫu nhiên. Chiết khấu `γ = 0.95` theo độ sâu.

**Multi-Restart**: Tối đa 10 restart × 3000 simulations/restart khi cây rơi vào dead-end.

> **Lưu ý**: MCTS dùng heuristic prior cố định — **không** có neural network. Đây là heuristic planning, không phải AlphaZero thật sự.

### 4.4 Reward Shaping

Bổ sung tín hiệu phần thưởng phụ để hướng dẫn agent học nhanh hơn khi reward terminal thưa:

**Phần thưởng terminal:**
- Thành công (còn lại = Target): `R = +100`
- Thất bại (còn lại ≠ Target): `R = -10 - |số_cuối - Target| / 100`

**Reward shaping bước trung gian:**

```
r(s, a, s') = -0.1  +  10 × (d(s) - d(s')) / max(1, Target)  +  2 × [result là ước Target]

với d(s) = min_{x ∈ s} |x - Target|
```

- `-0.1` — Chi phí bước đi, phạt kéo dài không cần thiết
- Thành phần giữa — Thưởng tiến trình: thưởng nếu gần Target hơn, phạt nếu xa hơn
- `+2` — Bonus chiến lược: ưu tiên tạo ước số của Target để dùng sau

### 4.5 Epsilon-Greedy Exploration

Cân bằng **exploration** (khám phá) và **exploitation** (khai thác):

```
Với xác suất ε:     Exploration — chọn hành động ngẫu nhiên (có hướng dẫn)
Với xác suất 1 - ε: Exploitation — chọn hành động có Q(s,a) cao nhất
```

Exploration trong dự án có **heuristic bias**: 60% chọn kết quả gần Target nhất, 40% ngẫu nhiên thuần.

**Epsilon decay:**
- Ban đầu: `ε = 0.45`
- Mỗi episode: `ε ← max(0.1, ε × 0.9997)`
- Tối thiểu: `ε_min = 0.1`

---

## 5. Tại sao RL? So sánh với các thuật toán khác

| Tiêu chí | BFS/DFS | A* Search | Backtracking | MCTS | Q-Learning |
|---|---|---|---|---|---|
| **Đảm bảo tìm nghiệm** | Có | Có | Có | Không | Không |
| **Hiệu quả bộ nhớ** | Kém | Trung bình | Tốt | Tốt | Tốt |
| **Tự cải thiện qua thời gian** | Không | Không | Không | Không | **Có** |
| **Tổng quát hóa sang puzzle mới** | Không | Không | Không | Không | **Có** |
| **Đo lường quá trình học** | Không | Không | Không | Hạn chế | **Có** |
| **Phù hợp nghiên cứu RL** | Không | Không | Không | Baseline | **Có** |

### Q-Learning vs. MCTS trong dự án

| Tiêu chí | Q-Learning | MCTS + PUCT |
|---|---|---|
| **Phân loại** | Model-Free RL | Model-Based Planning |
| **Học được gì** | Q(s,a) từ nhiều episode, lưu Q-table | Không học xuyên lần chạy |
| **Tổng quát hóa** | **Có** (state mang theo target) | Không |
| **Tốc độ giải 1 puzzle** | Chậm (cần nhiều episode) | Nhanh |
| **Phù hợp nghiên cứu RL** | **Có** | Baseline so sánh |
| **Hướng mở rộng** | DQN, Actor-Critic | AlphaZero thật sự |

---

## 6. Mô hình hóa bài toán Countdown dưới dạng MDP

```
Trạng thái ban đầu: s0 = (3, 6, 7, 8, 25, 50)   [sorted tuple]
Target: T = 952

Ví dụ một episode thành công:

s0 = (3, 6, 7, 8, 25, 50)  →  25 - 6 = 19   [r = shaping]
s1 = (3, 7, 8, 19, 50)     →   3 + 7 = 10   [r = shaping]
s2 = (8, 10, 19, 50)       →  19 x 50 = 950 [r = shaping]
s3 = (8, 10, 950)          →  10 - 8 = 2    [r = shaping]
s4 = (2, 950)              →  2 + 950 = 952 [r = +100, DONE]
s5 = (952,)                                 [terminal]
```

**Không gian trạng thái**: Hàng chục nghìn trạng thái khác nhau → phù hợp tabular Q-Learning, không cần function approximation.

**Hành động hợp lệ** (`get_valid_actions`):
- Chọn mọi cặp `(a, b)` phân biệt theo index.
- Phép toán: `+`, `×`, `a-b` (nếu a > b), `a÷b` (nếu b | a).
- Lọc trùng theo kết quả để tránh hành động thừa.

---

## 7. Kiến trúc hệ thống

```
+---------------------------------------------------------------+
|                        main.py                                 |
|   (Đọc input, chọn algo, điều phối, hiển thị, lưu kết quả)   |
+--------+------------------------------------------------------+
         |
   +-----+-----+
   |           |
   v           v
+----------+  +----------------------------------------------+
|countdown |  |               rl_agent.py                    |
|_env.py   |  |                                              |
|          |  |  +------------------+  +----------------+   |
| - State  |  |  |  MCTSRLSolver    |  | QLearningAgent |   |
| - Action |  |  |  (MCTS + PUCT)   |  | (TD Learning)  |   |
| - Reward |  |  +------------------+  +----------------+   |
| - P(s'|s)|  +----------------------------------------------+  +----------------------+
+----+----+           |                 GRPO (mới)  →  |  grpo_agent.py       |
     |                 v                                |  - PolicyNetwork     |
     |          +----------------+                      |    (PyTorch π(a|s,T))|
     +--------> |tree_visualizer |<- also used by GRPO  |  - generate_group    |
                | (DOT + PNG)    |   via recorder API   |  - verify_trajectory |
                +----------------+                      |  - GRPO loss/adv.    |
                                                        |  - train/eval/ckpt   |
                                                        +----------+-----------+
                                                                   v
                                              +--------------------------------+
                                              | grpo_checkpoint/ (policy.pt,    |
                                              | reference.pt, optimizer.pt,     |
                                              | meta.json) — tách biệt q_table  |
                                              +--------------------------------+
```

**Luồng chạy chính (`--mode solve`):**
1. `main.py` đọc `input.txt` → lấy `numbers` và `target`.
2. Khởi tạo `SearchTreeRecorder` để ghi cây tìm kiếm.
3. Gọi `solve_countdown_rl()` → agent chạy trong môi trường `CountdownEnv`.
4. Agent trả về chuỗi phép tính.
5. `main.py` dựng bước chi tiết, lưu `ketqua.txt`, xuất `tree.dot`/`tree.png`.
6. Hiển thị giao diện Matplotlib (trừ khi `--no-gui`).

---

## 8. Mô tả chi tiết từng file

### `countdown_env.py` — Môi trường MDP

**Vai trò**: Layer "physics" — định nghĩa cách trạng thái chuyển đổi và phần thưởng được tính.

**Class `CountdownEnv`**:

| Method | Mô tả |
|---|---|
| `__init__(numbers, target, require_use_all)` | Khởi tạo với bộ số, Target, luật dùng hết số |
| `reset()` | Đặt lại về s₀, trả về trạng thái ban đầu |
| `get_valid_actions(state)` | Liệt kê toàn bộ hành động hợp lệ tại `state` |
| `step_state(state, action)` | Thực hiện hành động, trả về `(next_state, reward, done, is_success)` |
| `step(action)` | Như `step_state` nhưng cập nhật `self.state` nội bộ |

**Kỹ thuật quan trọng**:
- Trạng thái là `tuple` sắp xếp tăng dần → normalize, tránh trùng lặp.
- Lọc cặp số trùng qua `seen_pairs`.
- `require_use_all=True`: kết thúc khi còn đúng **1 số**.
- `require_use_all=False` (`--allow-partial`): kết thúc ngay khi Target xuất hiện.

---

### `rl_agent.py` — Các thuật toán RL

**Vai trò**: Cài đặt MCTS và Q-Learning cho bài toán Countdown.

#### Class `MCTSNode`

Node cây MCTS lưu: `state`, `prior` P(s,a), `visits` N, `total_value`, `q_value` = total_value/visits, `untried_actions`, method `puct()`.

#### Class `MCTSRLSolver`

| Method | Mô tả |
|---|---|
| `__init__(...)` | Khởi tạo với môi trường, max_restarts, max_sims, c_puct |
| `compute_prior(state, action)` | Heuristic score (khoảng cách + bonus ước số) |
| `_get_actions_with_priors(state)` | Danh sách hành động kèm prior đã softmax |
| `search()` | MCTS đa-restart, trả về chuỗi phép tính và stats |
| `rollout(node)` | Guided rollout, trả về giá trị ước lượng |
| `backpropagate(node, value)` | Cập nhật N, Q từ node về gốc |
| `extract_path(node)` | Chuỗi phép tính từ gốc đến node nghiệm |

#### Class `QLearningAgent`

| Method | Mô tả |
|---|---|
| `__init__(numbers, target, α, γ, ε, ...)` | Khởi tạo với tham số học, tùy chọn nạp Q-table |
| `get_q(state, target, action)` | Tra cứu Q(s, target, a) |
| `choose_action(state, target, valid_actions, eps)` | Epsilon-greedy có heuristic bias |
| `train()` | Huấn luyện trên 1 instance, trả về lời giải đầu tiên |
| `train_batch(instances, episodes_per_instance)` | Huấn luyện đa instance, dùng chung Q-table |
| `greedy_success(numbers, target)` | Đánh giá policy greedy thuần trên puzzle |
| `save_q_table(path, instances)` | Lưu Q-table + metadata ra JSON |
| `load_q_table(path)` | Nạp Q-table từ JSON |

#### Hàm `generate_solvable_instances`

Sinh puzzle **đảm bảo có nghiệm dùng hết số**:
- Pool: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 100]`
- Áp phép tính ngẫu nhiên đến khi còn 1 số → Target.
- Chỉ nhận Target trong `[100, 999]`, trung gian ≤ 10000.
- `exclude` để tạo tập held-out không trùng train.

#### Hàm `solve_countdown_rl`

Entry point thống nhất: nhận `algorithm='mcts'`/`'qlearning'`, khởi tạo solver tương ứng.

---

### `main.py` — Điều phối chính

**Vai trò**: Điểm vào chương trình — đọc input, điều phối, hiển thị, lưu kết quả.

**Ba chế độ** (`--mode`):

| Chế độ | Lệnh | Mô tả |
|---|---|---|
| `solve` | `python main.py` | Giải puzzle trong `input.txt`, xuất lời giải + giao diện |
| `train` | `python main.py --mode train` | Huấn luyện Q-Learning đa instance, lưu Q-table (`--algo grpo` → huấn luyện GRPO) |
| `eval` | `python main.py --mode eval` | Đánh giá tổng quát hóa trên puzzle held-out (`--algo grpo` → eval GRPO) |

**Các hàm chính**:

| Hàm | Mô tả |
|---|---|
| `read_problem(path)` | Parse `input.txt` bằng regex, lấy numbers và target |
| `countdown_rl(numbers, target, ...)` | Wrapper gọi `solve_countdown_rl()` |
| `build_steps(numbers, operations)` | Dựng bước chi tiết + biểu thức toán học |
| `save_results(...)` | Lưu lời giải, stats vào `ketqua.txt` |
| `run_train(args)` | Sinh instance → huấn luyện → lưu Q-table |
| `run_eval(args)` | Nạp Q-table → đánh giá greedy → báo cáo |
| `show_simulation(...)` | Giao diện tương tác Matplotlib từng bước |
| `main()` | Parse CLI args, điều phối toàn bộ |

---

### `tree_visualizer.py` — Trực quan hóa cây

**Vai trò**: Ghi cây tìm kiếm RL (tối đa 2 tầng) ra Graphviz DOT, tùy chọn render PNG.

**Class `SearchTreeRecorder`**:

| Method | Mô tả |
|---|---|
| `start(state, visits, q_value)` | Tạo node gốc |
| `add_transition(parent_id, state, depth, op, ...)` | Thêm cạnh và node con |
| `finish_node(node_id, succeeded, visits, q_value)` | Đánh dấu SUCCESS/FAILED |
| `finish(succeeded, ...)` | Đánh dấu node gốc |
| `write_dot(dot_path)` | Xuất file `.dot` (Graphviz) |
| `render_png(dot_path, png_path)` | Gọi `dot` binary render PNG |

Mỗi node DOT hiển thị: trạng thái số, độ sâu, N, Q, status. Màu: xanh lá = SUCCESS, đỏ = FAILED, vàng = SEARCHING.

---

### `input.txt` — File đầu vào

```text
25 50 3 6 7 8
Target: 952
```
Dòng 1: 2–8 số nguyên dương. Dòng 2: `Target: <số>`.

---

### `ketqua.txt` — File đầu ra

Kết quả lần chạy gần nhất: thuật toán, số simulations/episodes, lời giải, từng bước, thời gian. Ghi đè mỗi lần chạy.

---

### `q_table.json` — Q-table đã học

Lưu từ `--mode train`:
```json
{
  "version": 1,
  "rules": {"require_use_all": true},
  "params": {"alpha": 0.25, "gamma": 0.95, "epsilon": 0.45},
  "train_instances": [[[3, 6, 7], 126]],
  "entries": [{"state": [3, 7, 25], "target": 952, "action": [25, "-", 3, 22], "q": 0.87}]
}
```

---

### `tree.dot` — Graphviz DOT

Cây tìm kiếm RL tối đa 2 tầng. Render:
```bash
dot -Tpng tree.dot -o tree.png
```

---

### `grpo_agent.py` — Thuật toán GRPO (RL thứ ba, thêm mới)

**Vai trò**: Phương pháp Policy-Gradient thứ ba, chạy song song MCTS/Q-Learning, **tái sử dụng nguyên vẹn MDP** trong `countdown_env.py` (State/Action/transition/reward không đổi). KHÔNG sửa `rl_agent.py`.

**Thành phần chính**:

| Hàm/Class | Mô tả |
|---|---|
| `encode_state_features` / `encode_action_features` / `encode_target_features` | Mã hóa state (multiset đổi độ dài), action, target thành tensor có số chiều cố định |
| `PolicyNetwork` (nn.Module) | π(a\|s, target) kiến trúc DeepSets: **chấm điểm từng cặp (state, action)** → logits đổi độ dài theo số hành động hợp lệ (action masking tự nhiên) |
| `verify_trajectory` | **Verifier độc lập** tái mô phỏng trajectory: mỗi số dùng ≤1 lần, không dùng số ngoại lai, không chia 0, phép chia phải hết, số học trung gian đúng, target đúng luật `require_use_all` |
| `calculate_trajectory_reward` | R(τ) cho GRPO (tách với env reward): +1 thành công đã verify, -1 không hợp lệ, còn lại -0.5..0 theo sai số |
| `generate_group` / `generate_trajectory` | Lấy mẫu **G trajectory** cho một puzzle từ π_old; lưu state/action/reward/old_log_prob/kết cục |
| `calculate_group_advantages` | A_i = (R_i − mean)/(std + ε); std=0 → trả 0 (không bịa advantage) |
| `calculate_grpo_loss` | Clipped surrogate + KL(π_θ‖π_ref) tùy chọn; tổng hợp log-prob **mức hành động** (mỗi bước = 1 token, chia theo độ dài τ) |
| `GRPOAgent.train_grpo` / `evaluate_grpo` / `save_checkpoint` / `load_checkpoint` | Vòng huấn luyện, đánh giá held-out, lưu/nạp checkpoint RIÊNG (không ghi đè `q_table.json`) |
| `make_data_splits` | Chia train/val/test **không giao nhau** từ `generate_solvable_instances` |

Checkpoint lưu tại thư mục `grpo_checkpoint/` (`policy.pt`, `reference.pt`, `optimizer.pt`, `meta.json`), nạp bằng `torch.load(weights_only=True)` — an toàn, không thực thi pickle tuple.

---

### `test_grpo.py` — Kiểm thử (36 test)

Unit test cho GRPO (mã hóa, verifier, advantage, loss hữu hạn + gradient + cập nhật tham số, checkpoint) và **hồi quy** cho MCTS/Q-Learning/CountdownEnv. Chạy: `py -m unittest test_grpo -v`.

---

### `benchmark.py` — So sánh trung thực các solver

Chạy trên **cùng bộ puzzle** (seed cố định): DFS cổ điển, MCTS+PUCT, Q-Learning, GRPO trước/sau huấn luyện (greedy + best-of-N). Đo exact solve rate, valid rate, thời gian, số bước, MAE. Mọi nghiệm đều qua `verify_trajectory`. Chạy: `py benchmark.py`.

---

## 9. Các thư viện được sử dụng

### Thư viện chuẩn Python (stdlib — không cần cài)

| Thư viện | File | Vai trò |
|---|---|---|
| `argparse` | `main.py` | Parse CLI: `--mode`, `--algo`, `--no-gui`, `--seed`, v.v. |
| `random` | `main.py`, `rl_agent.py` | Sinh ngẫu nhiên cho epsilon-greedy và MCTS rollout |
| `re` | `main.py` | Regex parse `input.txt` tách numbers và target |
| `sys` | `main.py` | `sys.exit()` khi lỗi; cấu hình UTF-8 cho Windows console |
| `time` | `main.py`, `rl_agent.py` | Đo thời gian bằng `time.perf_counter()` |
| `pathlib.Path` | Tất cả file | Quản lý đường dẫn cross-platform |
| `json` | `rl_agent.py` | Serialize/deserialize Q-table ra JSON |
| `math` | `rl_agent.py` | `sqrt()` cho PUCT, `exp()` cho softmax, `isclose()` so sánh float |
| `collections.defaultdict` | `rl_agent.py` | Q-table: tự trả 0.0 cho khóa chưa có |
| `collections.Counter` | `grpo_agent.py` | Verifier đếm multiset số còn dùng được |
| `dataclasses` | `grpo_agent.py` | `StepRecord`, `Trajectory`, `VerificationResult`, `GRPOConfig` |
| `shutil` | `tree_visualizer.py` | `which("dot")` kiểm tra Graphviz trong PATH |
| `subprocess` | `tree_visualizer.py` | Gọi `dot -Tpng` render cây ra PNG |
| `typing` | Tất cả file | Type hints: `Tuple`, `List`, `Dict`, `Any`, `Optional` |

### Thư viện bên ngoài (cần cài)

| Thư viện | File | Vai trò | Cài đặt |
|---|---|---|---|
| `matplotlib` | `main.py` | Giao diện tương tác: vẽ card số, nút điều hướng | `pip install matplotlib` |
| `matplotlib.patches.FancyBboxPatch` | `main.py` | Card số bo góc trong simulation | (theo matplotlib) |
| `matplotlib.widgets.Button` | `main.py` | Nút điều hướng bước giải | (theo matplotlib) |
| `torch` (PyTorch CPU) | `grpo_agent.py` | Mạng policy π(a\|s,target), autodiff cho GRPO loss, checkpoint | `pip install torch --index-url https://download.pytorch.org/whl/cpu` (đã kiểm chứng tương thích Python 3.14, torch 2.14.1+cpu) |

> **Lưu ý phụ thuộc**: `torch` CHỈ cần cho `--algo grpo`. Hai thuật toán cũ MCTS/Q-Learning chạy hoàn toàn bằng stdlib + matplotlib, KHÔNG bị ảnh hưởng nếu chưa cài torch (import trong `main.py` là lười — chỉ nạp khi dùng GRPO).

### Công cụ tùy chọn

| Công cụ | Mục đích |
|---|---|
| **Graphviz** (`dot`) | Render `tree.dot` → `tree.png` (file `.dot` luôn được tạo) |

---

## 10. Cài đặt và chạy chương trình

### Yêu cầu

- Python ≥ 3.8 (đã kiểm chứng thực tế trên **Python 3.14.6**)
- `matplotlib` (bắt buộc kể cả khi dùng `--no-gui`)
- `torch` CPU — **chỉ cần** khi dùng `--algo grpo` (MCTS/Q-Learning không cần)
- Graphviz (tùy chọn)

### Cài thư viện

```bash
pip install matplotlib
# Chỉ cần cho GRPO (bản CPU, nhẹ hơn bản CUDA):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Định dạng `input.txt`

```text
25 50 3 6 7 8
Target: 952
```

### Các lệnh chạy

```bash
# Giải bằng MCTS (mặc định) + giao diện
python main.py

# Giải bằng Q-Learning
python main.py --algo qlearning

# Không mở giao diện
python main.py --no-gui

# Cho phép dừng sớm (không cần dùng hết số)
python main.py --allow-partial

# Huấn luyện Q-Learning 100 instance × 150 episodes
python main.py --mode train --train-instances 100 --episodes-per-instance 150

# Đánh giá tổng quát hóa trên 50 puzzle held-out
python main.py --mode eval --eval-instances 50

# Seed cố định (tái hiện kết quả)
python main.py --mode train --seed 42

# ---- GRPO (cần cài torch) ----
# Huấn luyện GRPO rồi giải bằng policy đã học
python main.py --algo grpo --mode train --train-instances 150 --epochs 30 --group-size 8 --seed 7
python main.py --algo grpo --mode eval --eval-instances 20 --best-of-n 16
python main.py --algo grpo --mode solve --no-gui

# Benchmark so sánh DFS/MCTS/Q-Learning/GRPO trên cùng bộ puzzle
python benchmark.py --instances 20 --seed 999

# Chạy bộ kiểm thử (36 test: GRPO + hồi quy MCTS/Q-Learning)
python -m unittest test_grpo -v

# Render tree.png thủ công
dot -Tpng tree.dot -o tree.png
```

### Pipeline nghiên cứu đầy đủ

```bash
# 1. Huấn luyện
python main.py --mode train --train-instances 200 --episodes-per-instance 200 --seed 42

# 2. Đánh giá tổng quát hóa
python main.py --mode eval --eval-instances 100 --eval-seed 99

# 3. Giải puzzle từ Q-table đã học
python main.py --algo qlearning --no-gui
```

---

## 11. Ví dụ đầu ra

### Chế độ `solve` — MCTS

```
=== COUNTDOWN — REINFORCEMENT LEARNING ===
Các số: 25 50 3 6 7 8
Target: 952
Thuật toán: MCTS-RL (Monte Carlo Tree Search + PUCT)
Số lượt mô phỏng: 669
Root Visits: 669 | Root Q-Value: -27.57
Lời giải: (((3 + 7) - 8) + ((25 - 6) x 50)) = 952
Bước 1: 25 - 6 = 19
Bước 2:  3 + 7 = 10
Bước 3: 19 x 50 = 950
Bước 4: 10 - 8 = 2
Bước 5:  2 + 950 = 952
Thời gian: 59.005 ms
```

### Chế độ `train` — Q-Learning

```
Số instance: 100 | Episodes/instance: 150 | Tổng: 15000
Epsilon cuối: 0.1013
Số instance được giải khi train: 87/100
Số mục Q-table: 43821
Thời gian: 3241.7 ms
```

### Chế độ `eval` — Tổng quát hóa

```
Tỷ lệ success (held-out): 31/50 = 62.0%
Thời gian đánh giá: 12.3 ms
```

### Chế độ `eval` — GRPO (policy thuần, đã verifier)

```
[1] Policy greedy THUẦN (argmax π_θ):
    Exact solve rate:     0.0%        # trên test seed 999
    Valid solution rate:  100.0%
    Mean abs error:       22.6        # so với 200.3 trước huấn luyện
[2] Best-of-16 sampling:
    Exact solve rate:     5.0%
```

---

## 12. GRPO — Thuật toán RL thứ ba

**GRPO (Group Relative Policy Optimization)** là phương pháp Policy-Gradient dùng mạng neural làm policy được huấn luyện bằng gradient thật (PyTorch), bổ sung bên cạnh MCTS (planning) và Q-Learning (tabular TD). Module: `grpo_agent.py`.

### 12.1 Ánh xạ MDP (tái sử dụng CountdownEnv, không đổi cấu trúc)

| Thành phần | Cách GRPO xử lý |
|---|---|
| **State** `s` | tuple số còn lại + **target đưa vào policy** (π(a\|s, target)) vì target không nằm trong tuple → đủ tính Markov. Không đổi State để giữ tương thích Q-Learning/MCTS/visualization. |
| **Action** `a` | định dạng `(left, op, right, result)` của dự án. Số hành động **đổi theo state** → policy chấm điểm từng cặp (state, action), **không** giả định số chiều cố định. Xử lý số trùng: operation nhân bản theo multiset, trừ = lớn−bé>0, chia phải hết. |
| **P(s'\|s,a)** | môi trường **xác định**, P=1 cho next_state duy nhất; dùng lại `step_state`. Chính sách GRPO không làm đổi transition. |
| **Reward** `R` | **Tách bạch**: env reward theo bước (giữ nguyên cho MCTS/Q-Learning) ≠ trajectory reward R(τ) cho GRPO, tính từ **verifier độc lập**. |
| **γ (discount)** | GRPO **không dùng** γ của Q-Learning. Advantage tính trên R(τ) episode-level, group-relative, không discounted. |

### 12.2 Policy probability & action masking

`PolicyNetwork` (DeepSets): mỗi số → MLP → masked mean+max pooling (bất biến hoán vị, đổi độ dài được) ghép với embedding target → state_emb [D]; mỗi action ứng viên → action_emb [D]; `scorer(state_emb ⊕ action_emb)` → **1 logit/hành động**. `Categorical(logits)` chỉ trên danh sách hành động hợp lệ = **action masking tự nhiên** (hành động ngoài danh sách xác suất 0). Xác suất = softmax(logits); log-prob = `log_softmax`; lấy mẫu = `Categorical.sample()`.

### 12.3 Group-relative advantage & objective

Mỗi puzzle sinh **G trajectory** (một group). Với R(τ_i):
```
A_i = (R_i - mean_j R_j) / (std_j R_j + ε)      # std=0 → A=0 (không bịa advantage)
ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)               # importance ratio (π_old lưu lúc sinh)
L(θ) = mean_i [ -1/|τ_i| Σ_t min(ρ_t·A_i, clip(ρ_t, 1±ε)·A_i) ] + β·KL(π_θ‖π_ref)
```
Trajectory nhiều bước: mỗi bước = một "token" cùng chịu A_i cấp episode, loss chia theo độ dài |τ_i| (như paper GRPO). Cập nhật: Adam + clip grad-norm; test đã **chứng minh tham số đổi thật** và log-prob hành động tốt **tăng** sau update.

### 12.4 Dữ liệu & chi phí

`make_data_splits` chia train/val/test **không giao nhau** từ `generate_solvable_instances` (đã kiểm chứng disjoint trong test). Eval đo trên **đúng tập test** đã lưu trong checkpoint. Chi phí tham khảo: 150 puzzle × 30 epoch × G=8 ≈ 1912 lần cập nhật policy, cỡ **vài phút trên CPU**.

### 12.5 Kết quả benchmark trung thực (cùng 20 puzzle held-out, seed=999, require_use_all)

| Solver | Exact | Valid | Thời gian TB | Bước TB | MAE TB |
|---|---|---|---|---|---|
| DFS cổ điển (memo) | **20/20 = 100%** | 100% | 21.6 ms | 4.15 | 0.0 |
| MCTS + PUCT | **20/20 = 100%** | 100% | 203 ms | 4.15 | 0.0 |
| Q-Learning (greedy) | 1/20 = 5% | 100% | 0.8 ms | 4.15 | 144.6 |
| GRPO trước huấn luyện | 0/20 = 0% | 100% | 19.4 ms | 4.15 | 200.3 |
| GRPO sau huấn luyện (greedy) | 0/20 = 0% | 100% | 17.5 ms | 4.15 | **22.6** |
| GRPO sau huấn luyện (best-of-16) | 1/20 = 5% | 100% | 315.8 ms | 4.15 | 22.6 |

**Nhận định trung thực**: GRPO **có học thật** — MAE giảm mạnh 200 → 22.6 khi tổng hợp qua policy gradient; trên tập validation greedy đạt ~13.5% exact. **Tuy nhiên** trên không gian hữu hạn nhỏ của Countdown, GRPO **chưa vượt** MCTS/DFS (100%) về tỉ lệ nghiệm đúng tuyệt đối, và thời gian suy luận cộng chi phí huấn luyện **không** mang lại lợi thế. Đây là kết luận có bằng chứng, không khẳng định GRPO tốt hơn.

**Hạn chế ghi nhận**: sau ~30 epoch policy bị **sụp đổ entropy** (entropy ≈ 0.031, duplicate rate ≈ 81% trong group), khiến best-of-N mất đa dạng và gần bằng greedy.

### 12.6 Thí nghiệm KL chống sụp đổ entropy (β = 0.01, đã chạy thật)

Retrain từ đầu với `--kl-coeff 0.01` (KL(π_θ‖π_ref) với π_ref = snapshot khởi tạo, đã hỗ trợ sẵn trong `calculate_grpo_loss`, không sửa code), cấu hình còn lại giống hệt baseline (150 puzzle × 30 epoch, G=8, lr=0.002, seed=7 → **cùng train/val/test split**). Checkpoint: `grpo_checkpoint_kl/` (baseline giữ nguyên tại `grpo_checkpoint/`).

| Chỉ số | Baseline β=0 | KL β=0.01 |
|---|---|---|
| Entropy cuối / giữa (ep10) | 0.031 / 0.047 | **0.072 / 0.171** |
| Duplicate rate cuối | 81% | **74%** |
| Val exact (37 puzzle, epoch cuối) | 13.5% | **16.2%** |
| Held-out test greedy (20 puzzle seed 7) | **15.0%**, MAE 36.6 | 10.0%, MAE 36.8 |
| Held-out test best-of-16 | 15.0%, MAE 36.6 | 15.0%, MAE 36.6 |
| Benchmark greedy (20 puzzle seed 999) | 0/20, MAE **22.6** | 0/20, MAE 22.9 |
| Benchmark best-of-16 (seed 999) | 1/20 = 5%, MAE 22.6 | 1/20 = 5%, MAE 22.6 |

**Kết luận trung thực**: β=0.01 **làm chậm rõ rệt** sụp đổ entropy (ep10: 0.171 vs 0.047) và val_exact cuối nhỉnh hơn (16.2% vs 13.5%), nhưng **không ngăn được** collapse về cuối train (0.072/74%) và **không cải thiện đáng kể** khả năng giải held-out: MAE greedy gần như không đổi (22.9 vs 22.6 ở seed 999; 36.8 vs 36.6 ở test seed 7), exact rate kém hơn ở greedy test seed 7 (10% vs 15% — khác biệt 1/20 puzzle, trong biên độ nhiễu mẫu). Best-of-16 của KL chạy **tăng đúng 5pp** so với greedy của chính nó (15% vs 10%) trong khi baseline best-of-16 = greedy (15% vs 15%) — dấu hiệu đa dạng có hồi phục nhưng chưa đủ để vượt baseline. Muốn cải thiện thêm: β lớn hơn (0.03–0.1), entropy bonus, hoặc snapshot π_ref trượt; thời gian train tăng ~20% (1258 s so với baseline CPU tương đương).

---

## Ghi chú học thuật

- **Q-Learning `--algo qlearning`** có workflow hỗn hợp: dừng ngay khi tìm nghiệm đầu tiên. Để đánh giá đúng, dùng pipeline `train → eval`.
- **MCTS** dùng heuristic prior cố định → là **heuristic planning**, không phải AlphaZero.
- **Reward shaping** là progress-based, không phải potential-based chuẩn `γΦ(s') - Φ(s)` → không đảm bảo bảo toàn chính sách tối ưu lý thuyết.
- Để mở rộng sang **DQN** hoặc **AlphaZero thật sự**, cần neural network xấp xỉ Q-function hoặc policy/value network.
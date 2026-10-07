# Countdown Solver — mô hình hóa và hướng tiếp cận Reinforcement Learning

Dự án giải bài toán Countdown: từ một tập số, lần lượt chọn hai số và kết hợp
bằng `+`, `-`, `*`, `/` để đạt `Target`. Tài liệu này mô tả **đúng những gì code
đang làm**, phân biệt tìm kiếm với học tăng cường, và chọn một hướng lý thuyết
phù hợp nếu mục tiêu là xây dựng một thí nghiệm RL có thể đánh giá được.

## Tóm tắt và lựa chọn phương pháp

**Hướng RL được chọn: Q-Learning dạng bảng (tabular, model-free TD learning).**
Với một bài toán Countdown cố định, trạng thái và hành động là rời rạc, mỗi bước
làm giảm số lượng phần tử đúng một đơn vị, nên Q-Learning là phương pháp đơn
giản, dễ giải thích và phù hợp để làm mô hình RL cơ sở.

Code hiện có hai lựa chọn:

| Lựa chọn | Code thực hiện | Cách gọi đúng tên |
| --- | --- | --- |
| `qlearning` | Có Q-table và cập nhật TD/Bellman trên các trajectory chưa thành công. | Đây là nhánh có thành phần RL; xử lý episode thành công hiện dùng thêm return update riêng. |
| `mcts` (mặc định) | Tìm kiếm cây có giới hạn, PUCT, prior và rollout heuristic được viết thủ công. | Bộ giải tìm kiếm/lập kế hoạch heuristic; không phải AlphaZero và không tự học policy từ dữ liệu. |

Do đó, kết quả từ MCTS có thể hữu ích để tìm lời giải nhanh, nhưng số lần mô
phỏng hoặc Q-value trong cây **không tự chúng chứng minh agent đã học được chính
sách**. Q-Learning hiện cũng chỉ học trong bộ nhớ cho một lần chạy, trên đúng
bài toán đầu vào đó; chương trình dừng ngay khi tìm thấy episode thành công.
Chưa có lưu Q-table, tập đánh giá độc lập, hay bằng chứng tổng quát hóa sang
bài toán khác.

## Cấu trúc dự án

| Tệp | Vai trò |
| --- | --- |
| `main.py` | Đọc `input.txt`, chọn thuật toán, hiển thị/lưu lời giải và điều khiển giao diện Matplotlib. |
| `countdown_env.py` | Môi trường Countdown: trạng thái, hành động hợp lệ, chuyển trạng thái và reward. |
| `rl_agent.py` | Cài đặt tìm kiếm MCTS/PUCT heuristic và Q-Learning dạng bảng. |
| `tree_visualizer.py` | Ghi một phần cây/đường đi để xuất Graphviz DOT và PNG; đây là trực quan hóa, không phải mô hình policy đã học. |
| `input.txt` | Các số đầu vào và dòng `Target: <số>`. |
| `ketqua.txt` | Kết quả, thống kê và các phép tính của lần chạy gần nhất. |
| `tree.dot` | Biểu diễn DOT do lần chạy gần nhất sinh ra. |

## Đặc tả môi trường MDP trong code

Với một `Target` và luật chơi cố định, môi trường trong
[`countdown_env.py`](./countdown_env.py) có thể được mô tả bằng
\((S,A,P,R,\gamma)\). Vì mỗi phép tính đều xác định, chuyển trạng thái là **tất
định**; có thể viết \(P(s'|s,a)=1\) cho trạng thái kế tiếp hợp lệ.

### Trạng thái

Trạng thái \(s\) là tuple tăng dần của các số còn lại, ví dụ
\((3, 6, 7, 8, 25, 50)\). Đây là multiset được biểu diễn có thứ tự chuẩn hóa:
các số trùng nhau vẫn được giữ đủ số lần. Mỗi hành động loại hai phần tử và
thêm kết quả nên độ dài trạng thái giảm một.

`Target` không nằm trong tuple trạng thái. Điều này vẫn đủ Markov khi một môi
trường chỉ giải **một Target cố định**. Nếu dùng chung một Q-table cho nhiều
Target, cần đưa Target vào trạng thái (ví dụ `((numbers...), target)`) để trạng
thái chứa đủ thông tin quyết định reward và mục tiêu.

### Hành động và chuyển trạng thái

Một hành động được môi trường trả về dưới dạng
`(left, operator, right, result, next_state)`. Môi trường chọn một cặp số và
sinh kết quả nguyên dương:

- Cộng và nhân.
- Trừ số nhỏ khỏi số lớn, chỉ nhận kết quả dương.
- Chia nếu chia hết, không chia cho 0.

Trạng thái kế tiếp nhận các số không được chọn cùng với kết quả mới rồi sắp
xếp tăng dần. Hành động hợp lệ phụ thuộc trạng thái hiện tại \(A(s)\); không có
một tập hành động cố định cho mọi trạng thái.

### Điều kiện kết thúc và reward hiện tại

Với luật mặc định `require_use_all=True`, episode thành công khi phép tính làm
trạng thái còn đúng một số bằng Target; reward terminal là `+100`. Nếu còn đúng
một số khác Target, reward là `-10 - |số_cuối - Target| / 100`.

Với `--allow-partial`, đạt Target ở bất kỳ bước nào kết thúc episode với `+100`.
Khi chưa kết thúc, code dùng reward shaping:

\[
r(s,a,s')=-0.1+
10\frac{d(s)-d(s')}{\max(1,Target)}
+2\,\mathbf{1}[\text{result là ước của Target}],
\quad
d(s)=\min_{x\in s}|x-Target|.
\]

Đây là **shaping heuristic theo tiến triển**, không phải potential-based shaping
chuẩn: công thức hiện tại không dùng dạng
\(\gamma\Phi(s')-\Phi(s)\). Vì thế không nên khẳng định shaping này bảo toàn
chính sách tối ưu. Ngoài ra, `step_state` chưa đánh dấu `done=True` nếu trạng
thái có hơn một số nhưng không còn hành động hợp lệ; agent hiện tự thoát vòng
lặp trong trường hợp đó. Một môi trường RL hoàn chỉnh nên coi dead-end là
terminal rõ ràng.

## Phân tích hai thuật toán hiện thực

### 1. Q-Learning — thành phần RL chính được khuyến nghị

[`QLearningAgent`](./rl_agent.py) lưu giá trị
\(Q(s,a)\) theo cặp trạng thái/hành động và chọn hành động theo epsilon-greedy.
Các tham số mặc định trong code là:

- Learning rate \(\alpha=0.25\), discount \(\gamma=0.95\).
- Epsilon ban đầu `0.45`, giảm theo episode với hệ số `0.9997`, tối thiểu `0.1`.
- Tối đa `15000` episode.
- Với episode chưa thành công, cập nhật theo TD target
  \(r+\gamma\max_{a'}Q(s',a')\), bỏ phần bootstrap khi `done=True`.
- Khi tìm thấy lời giải, code cập nhật return `G` bắt đầu từ `100` ngược theo
  trajectory rồi **trả về ngay** lời giải đầu tiên; trajectory thành công này
  không đi qua vòng cập nhật TD thông thường ở cuối episode.

Một số chi tiết quan trọng để diễn giải đúng kết quả:

1. Nếu còn hai số và có hành động đạt Target, `choose_action` chọn hành động
   thắng ngay, không để policy Q-table quyết định. Đây là một heuristic hữu
   ích, nhưng một episode thắng không chứng minh Q-learning đã tự khám phá lời
   giải đó.
2. Trong bước thăm dò, 60% xác suất exploration chọn hành động có kết quả gần
   Target nhất; phần còn lại chọn ngẫu nhiên. Đây không phải exploration thuần
   ngẫu nhiên.
3. Bảng Q chỉ tồn tại trong đối tượng Python khi chạy. Nó không được lưu/đọc
   từ file, và `main.py` không có chế độ train rồi đánh giá policy đã lưu.
4. Vì trạng thái khởi đầu và Target cố định trong một lần chạy, kết quả hiện
   tại là giải một instance, không phải chứng cứ agent tổng quát hóa qua nhiều
   bộ số hoặc Target.
5. Bộ ghi cây cho Q-Learning ghi đường đi tìm được (tối đa hai tầng), không ghi
   đầy đủ tất cả episode hay policy đã học.

Vì vậy, nhánh Q-Learning hiện là **cài đặt RL nền tảng nhưng có workflow lai**:
TD update cho các episode chưa thành công và return update riêng cho episode
thành công. Chương trình vẫn trộn huấn luyện, tìm lời giải và dừng sớm. Để đánh
giá học được gì, cần tách huấn luyện khỏi đánh giá và chạy policy greedy trên
các episode chưa dùng trong huấn luyện.

### 2. MCTS/PUCT — bộ giải tìm kiếm heuristic

`MCTSRLSolver` xây cây từ trạng thái ban đầu, chọn nhánh bằng công thức PUCT,
và lan truyền giá trị rollout về các node. Tuy nhiên:

- Prior \(P(s,a)\) là softmax của điểm heuristic dựa trên khoảng cách từ **kết
  quả của hành động** tới Target và bonus ước số; nó không được học từ dữ liệu.
- Rollout chọn hành động gần Target nhất với xác suất `0.7`, nếu không thì
  chọn ngẫu nhiên; rollout chiết khấu reward với `0.95`.
- Cây bị khởi tạo lại ở mỗi restart. Mặc định có tối đa 10 restart, mỗi lần tối
  đa 3000 simulations. Đây là giới hạn tìm kiếm, không phải bảo đảm tìm ra
  nghiệm hay bảo đảm hội tụ.
- Không có policy/value neural network, cập nhật trọng số, replay/training
  dataset hay cơ chế tự cải thiện xuyên lần chạy. Vì vậy không nên mô tả nó là
  “AlphaZero-style learning”. PUCT là công thức lựa chọn trong tìm kiếm; bản
  thân nó không biến heuristic search thành thuật toán RL.

Nên giữ MCTS như baseline tìm kiếm/planning để so sánh thời gian và tỷ lệ tìm
được lời giải. Nếu muốn triển khai AlphaZero thật sự, cần policy/value model
được huấn luyện trên dữ liệu kết quả tìm kiếm; đó là một hướng khác, phức tạp
hơn và không phải code hiện tại.

## Hướng triển khai RL được đề xuất

Để có một thí nghiệm phù hợp với RL và có thể bảo vệ được về mặt học thuật,
nên phát triển Q-Learning theo các bước sau:

1. **Chốt MDP:** Dùng trạng thái `(multiset_numbers, target)` nếu có nhiều bài
   toán; xác định rõ terminal success, terminal failure/dead-end và luật
   `require_use_all`.
2. **Chuẩn hóa episode:** Gọi `reset()` đầu episode; mỗi bước lấy `A(s)`, chọn
   hành động, nhận `(s', r, done)`. Nếu terminal thì đặt giá trị bootstrap bằng
   0. Không coi trạng thái không có hành động là trạng thái tiếp tục.
3. **Dùng cập nhật Q-Learning nhất quán:**

   \[
   Q(s,a)\leftarrow Q(s,a)+\alpha
   \left[r+\gamma(1-done)\max_{a'\in A(s')}Q(s',a')-Q(s,a)\right].
   \]

   Khi đã có đủ dữ liệu, đánh giá policy greedy riêng; không dừng quá trình
   huấn luyện ngay ở episode thắng đầu tiên nếu mục tiêu là đo quá trình học.
4. **Thiết kế reward có chủ đích:** Giữ reward terminal làm tín hiệu chính.
   Nếu cần shaping mà muốn giữ tối ưu hóa tương đương, chọn potential \(\Phi\)
   và dùng \(r'=r+\gamma\Phi(s')-\Phi(s)\); kiểm tra riêng reward khi thành
   công, thất bại, dead-end và chế độ partial.
5. **Đánh giá tách biệt:** Sinh/tách tập puzzle train và test; đo success rate,
   số episode tới lời giải, return, độ dài lời giải và thời gian. Báo cáo nhiều
   seed hoặc nhiều puzzle, so với MCTS và một baseline tìm kiếm chính xác nếu
   có. Một lần chạy ra đúng Target chỉ kiểm chứng lời giải, không đo được
   learning performance.
6. **Lưu artifact học:** Lưu Q-table cùng phiên bản luật/tham số, rồi nạp lại
   trong chế độ evaluation. Với một bài toán nhỏ cố định, tabular Q-learning
   dễ kiểm tra; nếu mục tiêu là tổng quát hóa trên nhiều puzzle và bảng quá
   lớn, khi đó mới cân nhắc function approximation như DQN.

Các mục trên là **định hướng đề xuất**, chưa được triển khai đầy đủ trong code
hiện tại.

## Cài đặt và chạy

Yêu cầu thư viện ngoài: `matplotlib` (được import ngay cả khi dùng `--no-gui`).
Graphviz là tùy chọn, chỉ cần nếu muốn render `tree.png`; file `tree.dot` vẫn
được ghi khi không có Graphviz.

```bash
python -m pip install matplotlib
```

Định dạng `input.txt`: nhập 2–8 số nguyên dương trước dòng Target:

```text
25 50 3 6 7 8
Target: 952
```

Chạy mặc định dùng MCTS/PUCT heuristic và mở giao diện:

```bash
python main.py
```

Chạy Q-Learning hoặc tắt giao diện:

```bash
python main.py --algo qlearning
python main.py --no-gui
python main.py --algo qlearning --no-gui
```

Cho phép dừng ngay khi đạt Target mà không cần dùng hết số:

```bash
python main.py --allow-partial
```

`--no-gui` chỉ tắt cửa sổ Matplotlib, không bỏ yêu cầu cài Matplotlib. Các tùy
chọn `--algo` và `--allow-partial` được xử lý trong [`main.py`](./main.py).

## Ví dụ đầu ra

Với input ở trên, kết quả đã được ghi trong `ketqua.txt` của workspace tại thời
điểm cập nhật tài liệu:

```text
Thuật toán: MCTS-RL (Monte Carlo Tree Search + PUCT)
Số lượt mô phỏng (Simulations): 669
Root Visits: 669 | Root Q-Value: -27.57
Lời giải: (((3 + 7) - 8) + ((25 - 6) * 50)) = 952
Bước 1: 25 − 6 = 19
Bước 2: 3 + 7 = 10
Bước 3: 19 × 50 = 950
Bước 4: 10 − 8 = 2
Bước 5: 2 + 950 = 952
Thời gian tìm kiếm: 59.005 ms
```

Đây là một kết quả cụ thể của một lần chạy, không phải số liệu ổn định hay cam
kết MCTS luôn tìm được lời giải. Kết quả có thể thay đổi do rollout có ngẫu
nhiên; hãy đánh giá trên nhiều instance/lần chạy trước khi kết luận hiệu năng.

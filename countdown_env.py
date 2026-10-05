"""
countdown_env.py — Môi trường MDP (Markov Decision Process) cho bài toán Countdown.

Mô hình hóa bài toán Countdown dưới dạng bài toán Học tăng cường (Reinforcement Learning):
- Không gian trạng thái (State space):
    Trạng thái s là một tuple các số nguyên dương hiện có, được sắp xếp tăng dần.
- Không gian hành động (Action space):
    Mỗi hành động a chọn 2 phần tử (a, b) từ trạng thái s và một phép toán nhị phân (+, -, *, /)
    thỏa mãn luật chơi Countdown:
    + Phép cộng: a + b
    + Phép nhân: a * b
    + Phép trừ: max(a, b) - min(a, b) > 0 (kết quả nguyên dương)
    + Phép chia: max % min == 0 (chia hết, kết quả nguyên dương, không chia cho 0)
- Hàm chuyển trạng thái P(s' | s, a):
    Loại bỏ a và b khỏi s, thêm kết quả c vào, sắp xếp lại thành s'.
- Hàm phần thưởng R(s, a, s'):
    + Terminal Thành công: +100 khi đạt target (và dùng hết số nếu require_use_all=True)
    + Terminal Thất bại: Phạt -10 kèm độ lệch so với target
    + Reward Shaping: Thưởng tiến trình giảm khoảng cách tới target (Potential-based),
      cộng điểm thưởng nếu tạo ra ước số của target, kèm chi phí bước đi (-0.1).
"""

from typing import Tuple, List, Dict, Any, Optional


class CountdownEnv:
    """Môi trường mô phỏng trò chơi Countdown phục vụ huấn luyện RL."""

    def __init__(self, numbers: List[int], target: int, require_use_all: bool = True):
        self.initial_numbers = tuple(sorted(numbers))
        self.target = target
        self.require_use_all = require_use_all
        self.state = self.initial_numbers

    def reset(self) -> Tuple[int, ...]:
        """Khởi động lại môi trường về trạng thái ban đầu."""
        self.state = self.initial_numbers
        return self.state

    def get_valid_actions(self, state: Optional[Tuple[int, ...]] = None) -> List[Tuple[int, str, int, int, Tuple[int, ...]]]:
        """
        Trả về danh sách các hành động hợp lệ từ trạng thái hiện tại.
        Mỗi hành động là một tuple:
            (left_operand, operator, right_operand, result, next_state)
        """
        if state is None:
            state = self.state

        if len(state) < 2:
            return []

        actions = []
        n = len(state)
        seen_pairs = set()

        for i in range(n):
            for j in range(i + 1, n):
                a, b = state[i], state[j]
                pair_key = (a, b)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                rem_values = list(state)
                del rem_values[j]
                del rem_values[i]
                rem = tuple(rem_values)

                # Tập các ứng viên phép toán
                candidates = [
                    (a, "+", b, a + b),
                    (a, "*", b, a * b),
                ]
                if a > b:
                    candidates.append((a, "-", b, a - b))
                elif b > a:
                    candidates.append((b, "-", a, b - a))

                if b != 0 and a % b == 0:
                    candidates.append((a, "/", b, a // b))
                if a != 0 and b % a == 0 and a != b:
                    candidates.append((b, "/", a, b // a))

                tried_results = set()
                for left, op, right, res in candidates:
                    if res <= 0 or res in tried_results:
                        continue
                    tried_results.add(res)
                    next_s = tuple(sorted(rem + (res,)))
                    actions.append((left, op, right, res, next_s))

        return actions

    def step_state(
        self, state: Tuple[int, ...], action: Tuple[int, str, int, int, Tuple[int, ...]]
    ) -> Tuple[Tuple[int, ...], float, bool, bool]:
        """
        Thực hiện một bước chuyển trạng thái (deterministic transition).
        Trả về: (next_state, reward, done, is_success)
        """
        left, op, right, res, next_state = action
        done = False
        is_success = False

        if self.require_use_all:
            # Luật Countdown chuẩn của dự án: phải dùng hết các số (chỉ còn 1 số cuối cùng)
            if len(next_state) == 1:
                done = True
                if next_state[0] == self.target:
                    reward = 100.0
                    is_success = True
                else:
                    dist = abs(next_state[0] - self.target)
                    reward = -10.0 - (dist / 100.0)
            else:
                # Reward shaping: tiến trình tới target
                curr_dist = min(abs(x - self.target) for x in state)
                next_dist = min(abs(x - self.target) for x in next_state)
                progress = curr_dist - next_dist
                div_bonus = 2.0 if (res > 1 and self.target % res == 0) else 0.0
                reward = -0.1 + (progress / max(1, self.target)) * 10.0 + div_bonus
        else:
            # Cho phép dừng ngay khi đạt target ở bất kỳ bước nào
            if res == self.target or self.target in next_state:
                reward = 100.0
                done = True
                is_success = True
            elif len(next_state) == 1:
                done = True
                dist = abs(next_state[0] - self.target)
                reward = -10.0 - (dist / 100.0)
            else:
                curr_dist = min(abs(x - self.target) for x in state)
                next_dist = min(abs(x - self.target) for x in next_state)
                progress = curr_dist - next_dist
                div_bonus = 2.0 if (res > 1 and self.target % res == 0) else 0.0
                reward = -0.1 + (progress / max(1, self.target)) * 10.0 + div_bonus

        return next_state, reward, done, is_success

    def step(
        self, action: Tuple[int, str, int, int, Tuple[int, ...]]
    ) -> Tuple[Tuple[int, ...], float, bool, bool]:
        """Thực hiện hành động trên trạng thái hiện tại của môi trường."""
        next_state, reward, done, is_success = self.step_state(self.state, action)
        self.state = next_state
        return next_state, reward, done, is_success

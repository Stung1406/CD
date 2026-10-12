"""
grpo_agent.py — Thuật toán GRPO (Group Relative Policy Optimization) cho Countdown.

PHƯƠNG PHÁP RL THỨ BA của dự án, song song với MCTS+PUCT và Q-Learning.
Hai thuật toán cũ trong `rl_agent.py` và môi trường `countdown_env.py` KHÔNG bị
sửa đổi; GRPO tái sử dụng trực tiếp MDP hiện có.

Tái sử dụng MDP từ `countdown_env.CountdownEnv`:
- STATE   : tuple các số còn lại (sắp xếp tăng dần). Target KHÔNG nằm trong tuple
            nên chính sách được định nghĩa là π(a | s, target) — cả state lẫn
            target đều được đưa vào mạng policy (tính Markov đầy đủ). Không đổi
            cấu trúc State của dự án để Q-Learning / MCTS / visualization còn chạy.
- ACTION  : định dạng chuẩn dự án (left, op, right, result[, next_state]).
            Số hành động hợp lệ THAY ĐỔI theo trạng thái, vì vậy policy là mô
            hình CHẤM ĐIỂM TỪNG CẶP (state, action) + action masking (softmax chỉ
            trên danh sách hành động hợp lệ), không dùng đầu ra cố định.
- P       : môi trường xác định, P(s'|s,a) = 1; dùng lại CountdownEnv.step_state.
            GRPO không làm thay đổi transition.
- R       : TÁCH BẠCH — env reward theo bước (giữ nguyên cho MCTS/Q-Learning)
            và trajectory reward R(τ) cho GRPO tính từ VERIFIER ĐỘC LẬP:
            +1.0 thành công đã kiểm chứng; -1.0 hành động không hợp lệ;
            còn lại -0.5 + 0.5·credit với credit = 1 - min_error/target.
- GAMMA   : GRPO KHÔNG dùng gamma của Q-Learning. Advantage tính trên reward
            episode-level của cả trajectory (group-relative), không discounted.

OBJECTIVE (action-level; mỗi bước t trong trajectory xem như một "token"):
    ρ_t   = π_θ(a_t | s_t, target) / π_old(a_t | s_t, target)   (importance ratio)
    A_i   = (R(τ_i) - mean_j R(τ_j)) / (std_j R(τ_j) + ε_adv)   (group-relative)
    L(θ)  = mean_i [ -1/|τ_i| Σ_t min(ρ_t·A_i, clip(ρ_t, 1±ε)·A_i) ]
            + β · mean_t KL(π_θ(·|s_t, target) ‖ π_ref(·|s_t, target))
π_old là snapshot tại lúc sinh (log-prob đã lưu); π_ref bất động, chỉ dùng khi
β > 0. Gradient backprop qua log π_θ tính lại mỗi inner-epoch, Adam optimizer,
clip tổng norm gradient. Mặc định chạy CPU (~50k tham số).
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
from torch.distributions import Categorical

from countdown_env import CountdownEnv
from rl_agent import generate_solvable_instances

ActionKey = Tuple[int, str, int, int]          # (left, op, right, result) — 4-tuple chuẩn của build_steps
State = Tuple[int, ...]

MAX_SLOTS = 8                                   # input.txt cho phép tối đa 8 số
OP_LIST = ("+", "-", "*", "/")
LOG_DENOM = math.log(10001.0)


# ---------------------------------------------------------------------------
# 1. MÃ HÓA STATE / ACTION — biến độ dài thay đổi thành tensor cố định
# ---------------------------------------------------------------------------

def encode_state_features(state: State, target: int, max_slots: int = MAX_SLOTS) -> torch.Tensor:
    """
    Mã hóa state (multiset số còn lại, độ dài 1..max_slots) + target.

    Output: tensor [max_slots, 5] float32; slot thừa zero-padding (mask = chiều 4).
    Feature mỗi số: [value/1000, log1p(value)/log(10001),
                    1 - min(|value-target|,target)/target,
                    1 nếu value>1 và value chia hết target, mask_slot].
    """
    feats: List[List[float]] = []
    for value in state[:max_slots]:
        dist = min(abs(value - target), abs(target))
        feats.append([
            value / 1000.0,
            math.log1p(value) / LOG_DENOM,
            1.0 - dist / max(1, abs(target)),
            1.0 if (value > 1 and target % value == 0) else 0.0,
            1.0,
        ])
    while len(feats) < max_slots:
        feats.append([0.0] * 5)
    return torch.tensor(feats, dtype=torch.float32)


def encode_action_features(actions: Sequence[ActionKey], target: int) -> torch.Tensor:
    """
    Mã hóa danh sách hành động ứng viên (độ dài n BIẾN THIÊN theo trạng thái).

    Output: tensor [n_actions, 9] float32.
    Feature: [left/1000, right/1000, result/1000, log1p(result)/log(10001),
              one-hot(+,-,*,/), 1 - min(|result-target|,target)/target].
    """
    rows: List[List[float]] = []
    for left, op, right, res in actions:
        onehot = [1.0 if op == name else 0.0 for name in OP_LIST]
        dist = min(abs(res - target), abs(target))
        rows.append([left / 1000.0, right / 1000.0, res / 1000.0,
                     math.log1p(res) / LOG_DENOM, *onehot,
                     1.0 - dist / max(1, abs(target))])
    return torch.tensor(rows, dtype=torch.float32)


def encode_target_features(target: int) -> torch.Tensor:
    """Mã hóa target: tensor [4] = [t/1000, log1p(t)/log(10001), chữ_số/4, parity]."""
    return torch.tensor(
        [target / 1000.0, math.log1p(target) / LOG_DENOM, len(str(target)) / 4.0, float(target % 2)],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# 2. POLICY MODEL π(a | s, target) — DeepSets scorer, PyTorch
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Chính sách action-level π(a | s, target) (~50k tham số, CPU-friendly).

    Kiến trúc:
      NumberEncoder : MLP 5→32→D trên TỪNG số của state, gộp masked mean+max
                      pooling (DeepSets) => bất biến hoán vị, độ dài biến thiên.
      TargetEncoder : MLP 4→D.
      StateEmbed    : concat(pooled, tgt_emb) → MLP 2D→D (target đã tích hợp).
      ActionEncoder : MLP 9→32→D.
      Scorer        : MLP concat(state_emb ⊕ action_emb)[2D] → 32 → 1 = logit.

    logits = forward(state, target, valid_actions)      # [n_valid]
    dist   = Categorical(logits=logits)  → softmax CHỈ trên hành động hợp lệ
               (action masking tự nhiên: không có trong danh sách ⇒ xác suất 0)
    prob   = dist.probs;  sample = dist.sample();  log_prob = dist.log_prob(idx)
    backprop: trong calculate_grpo_loss logits được tính LẠI bằng π_θ (giữ
    computation graph); loss.backward() + optimizer.step() cập nhật tham số thật.
    Không có bất kỳ heuristic cứng nào đóng vai policy đã huấn luyện.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_enc = nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, d_model), nn.ReLU())
        self.tgt_enc = nn.Sequential(nn.Linear(4, d_model), nn.ReLU())
        self.state_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.act_enc = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, d_model), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(2 * d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def state_embedding(self, state: State, target: int) -> torch.Tensor:
        """(state, target) → tensor [D] = state_emb (ĐÃ tích hợp thông tin target)."""
        feats = encode_state_features(state, target)              # [MAX_SLOTS, 5]
        mask = feats[:, 4:5]                                      # [MAX_SLOTS, 1]
        emb = self.num_enc(feats)                                 # [MAX_SLOTS, D]
        mean_pool = (emb * mask).sum(0) / mask.sum(0).clamp(min=1.0)
        masked = emb.masked_fill(mask == 0, float("-inf"))
        max_pool = torch.nan_to_num(masked.max(0).values, neginf=0.0)
        tgt = self.tgt_enc(encode_target_features(target))         # [D]
        return self.state_mlp(torch.cat([mean_pool + max_pool, tgt], dim=0))  # [D]

    def forward(self, state: State, target: int, actions: Sequence[ActionKey]) -> torch.Tensor:
        """
        Logit cho từng hành động hợp lệ của trạng thái.

        Input: state tuple, target int, actions dãy độ dài n (biến thiên).
        Output: tensor [n] float (chưa chuẩn hóa). n = đúng số hành động hợp lệ.
        """
        s_emb = self.state_embedding(state, target)                # [D]
        a_feats = encode_action_features(actions, target)          # [n, 9]
        a_emb = self.act_enc(a_feats)                              # [n, D]
        paired = torch.cat([s_emb.unsqueeze(0).expand(a_emb.shape[0], -1), a_emb], dim=1)  # [n, 2D]
        return self.score(paired).squeeze(-1)                      # [n]


# ---------------------------------------------------------------------------
# 3. VERIFIER ĐỘC LẬP — kiểm chứng trajectory (không so sánh text)
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Kết quả kiểm chứng độc lập: hợp lệ? thành công? lý do; số cuối; sai số nhỏ nhất."""
    valid: bool
    success: bool
    reason: str
    final_value: Optional[int]
    min_error: Optional[int]


def verify_trajectory(
    initial_numbers: Sequence[int],
    target: int,
    actions: Sequence[ActionKey],
    require_use_all: bool = True,
) -> VerificationResult:
    """
    Verifier ĐỘC LẬP tái mô phỏng trajectory trên multiset số ban đầu:
      1. Mỗi số dùng đúng số lần có trong multiset (kể cả kết quả trung gian);
         operand không tồn tại ⇒ invalid (không dùng được số ngoài tập).
      2. Không chia cho 0; phép chia phải chia hết, kết quả nguyên dương.
      3. Phép trừ phải cho kết quả nguyên dương: left - right == result > 0.
      4. Kết quả mọi phép tính được TÍNH LẠI đúng số học (không tin chuỗi text).
      5. require_use_all=True: thành công ⇔ còn đúng 1 số và bằng target.
         require_use_all=False: thành công ⇔ xuất hiện target ở bất kỳ bước nào
         (hoặc số ban đầu đã bằng target).
    success=True luôn kéo theo valid=True.
    """
    pool: Counter = Counter(int(x) for x in initial_numbers)
    min_error: Optional[int] = None
    for value in pool:
        err = abs(value - target)
        min_error = err if min_error is None else min(min_error, err)
    if not require_use_all and pool.get(target, 0) > 0:
        return VerificationResult(True, True, "success", target, 0)

    for left, op, right, res in actions:
        left, right, res = int(left), int(right), int(res)
        if pool[left] <= 0 or pool[right] <= 0 or (left == right and pool[left] < 2):
            return VerificationResult(False, False, f"invalid:operand {left} or {right} not available",
                                      None, min_error)
        pool[left] -= 1
        pool[right] -= 1
        if op == "+":
            expected = left + right
        elif op == "-":
            expected = left - right
            if expected <= 0:
                return VerificationResult(False, False, "invalid:non-positive subtraction", None, min_error)
        elif op == "*":
            expected = left * right
        elif op == "/":
            if right == 0 or left % right != 0 or left // right <= 0:
                return VerificationResult(False, False, "invalid:illegal division", None, min_error)
            expected = left // right
        else:
            return VerificationResult(False, False, f"invalid:unknown op {op}", None, min_error)
        if expected != res:
            return VerificationResult(False, False, f"invalid:bad arithmetic {left}{op}{right}!={res}",
                                      None, min_error)
        pool[res] += 1
        err = abs(res - target)
        min_error = err if min_error is None else min(min_error, err)
        if not require_use_all and res == target:
            return VerificationResult(True, True, "success", res, 0)

    remaining = sum(pool.values())
    final_value = next(iter(pool.elements())) if remaining == 1 else None
    if require_use_all:
        if remaining == 1 and final_value == target:
            return VerificationResult(True, True, "success", final_value, 0)
        if remaining == 1:
            return VerificationResult(True, False, "wrong_final", final_value, min_error)
        return VerificationResult(True, False, "not_reduced_to_one", final_value, min_error)
    return VerificationResult(True, False, "target_not_reached", final_value, min_error)


def calculate_trajectory_reward(
    initial_numbers: Sequence[int],
    target: int,
    actions: Sequence[ActionKey],
    require_use_all: bool = True,
) -> Tuple[float, VerificationResult]:
    """
    Reward trajectory R(τ) cho GRPO (tách biệt env reward theo bước):
      +1.0  verifier xác nhận thành công đúng luật;
      -1.0  trajectory không hợp lệ (hành động bịa / sai số học);
      khác  -0.5 + 0.5·credit, credit = max(0, 1 - min_error/target) — liên tục
            theo sai số, giúp std của nhóm hiếm khi bằng 0 một cách tầm thường.
    """
    result = verify_trajectory(initial_numbers, target, actions, require_use_all)
    if result.success:
        return 1.0, result
    if not result.valid:
        return -1.0, result
    credit = max(0.0, 1.0 - (result.min_error if result.min_error is not None else target) / max(1, abs(target)))
    return -0.5 + 0.5 * credit, result


# ---------------------------------------------------------------------------
# 4. GENERATE GROUP / TRAJECTORY
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Một bước trajectory, đủ để tái tính logπ_θ, ratio và KL."""
    state: State
    target: int
    actions: Tuple[ActionKey, ...]   # mọi hành động hợp lệ tại state (thứ tự tất định)
    taken_idx: int                   # index hành động đã chọn
    old_log_prob: float              # log π_old(a_t | s_t, target) lúc sinh


@dataclass
class Trajectory:
    """Một trajectory = một chuỗi hành động hoàn chỉnh cho MỘT bài toán.
    (KHÔNG nhầm với group_size: group là NHIỀU trajectory như vậy.)"""
    steps: List[StepRecord] = field(default_factory=list)
    actions: List[ActionKey] = field(default_factory=list)
    env_reward: float = 0.0          # tổng env reward (chỉ để tham khảo)
    reward: float = 0.0              # R(τ) từ verifier — dùng cho advantage
    success: bool = False
    valid: bool = False
    reason: str = ""
    final_value: Optional[int] = None
    min_error: Optional[int] = None
    entropy_mean: float = 0.0

    @property
    def key(self) -> Tuple[ActionKey, ...]:
        """Khóa nhận dạng trajectory (phát hiện trùng lặp trong group)."""
        return tuple(self.actions)


@dataclass
class GRPOConfig:
    """Siêu tham số GRPO (mặc định hướng CPU; mọi tên khớp argparse của main.py)."""
    learning_rate: float = 3e-3
    batch_size: int = 4                 # số puzzle mỗi lần sinh group + cập nhật
    group_size: int = 8                 # G trajectory mỗi puzzle (KHÔNG phải số bước)
    num_epochs: int = 10                # số lần đi qua toàn bộ train split
    inner_epochs: int = 2               # PPO-style tái dùng batch đã sinh
    max_steps_per_episode: int = 16     # trần an toàn (thực tế ≤ k-1 bước)
    clip_epsilon: float = 0.2           # ε clipped surrogate
    kl_coefficient: float = 0.0         # β KL(π_θ‖π_ref); 0 = tắt
    adv_epsilon: float = 1e-4           # ε mẫu số advantage
    seed: Optional[int] = None
    checkpoint_interval: int = 2        # lưu checkpoint mỗi N epoch
    evaluation_interval: int = 2        # val-set eval mỗi N epoch
    device: str = "cpu"
    max_gradient_norm: float = 1.0
    d_model: int = 64
    require_use_all: bool = True

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


def generate_trajectory(
    policy: PolicyNetwork,
    numbers: Sequence[int],
    target: int,
    config: GRPOConfig,
    greedy: bool = False,
) -> Trajectory:
    """
    Sinh MỘT trajectory từ chính sách hiện hành (sample) hoặc argmax (greedy).

    Mỗi bước: lấy hành động hợp lệ từ CountdownEnv.get_valid_actions, tính logits
    (no_grad), mẫu Categorical, lưu StepRecord(state, target, actions, idx,
    old_log_prob) rồi chuyển tiếp bằng CountdownEnv.step_state (P xác định).
    Kết cục do verify_trajectory + calculate_trajectory_reward quyết định —
    KHÔNG tin cờ done/is_success của env.
    """
    env = CountdownEnv(list(numbers), target, require_use_all=config.require_use_all)
    state = env.reset()
    traj = Trajectory()
    entropies: List[float] = []
    for _ in range(config.max_steps_per_episode):
        if len(state) < 2:
            break
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            traj.reason = "no_valid_actions"
            break
        keys = tuple((a[0], a[1], a[2], a[3]) for a in valid_actions)
        with torch.no_grad():
            logits = policy(state, target, keys)
            dist = Categorical(logits=logits)
            idx = int(torch.argmax(logits)) if greedy else int(dist.sample())
            log_prob = float(dist.log_prob(torch.tensor(idx)))
            entropies.append(float(dist.entropy()))
        action = valid_actions[idx]
        next_state, reward, done, _ = env.step_state(state, action)
        traj.steps.append(StepRecord(state, target, keys, idx, log_prob))
        traj.actions.append(keys[idx])
        traj.env_reward += reward
        state = next_state
        if done:
            break
    if not traj.reason:
        traj.reason = "reached_final" if len(state) == 1 else "max_steps"
    verification = verify_trajectory(list(numbers), target, traj.actions, config.require_use_all)
    traj.reward, _ = calculate_trajectory_reward(list(numbers), target, traj.actions, config.require_use_all)
    traj.success = verification.success
    traj.valid = verification.valid
    traj.final_value = verification.final_value
    traj.min_error = verification.min_error
    traj.entropy_mean = sum(entropies) / len(entropies) if entropies else 0.0
    if verification.success:
        traj.reason = "success"
    return traj


def generate_group(
    policy: PolicyNetwork,
    numbers: Sequence[int],
    target: int,
    config: GRPOConfig,
) -> List[Trajectory]:
    """Generate group: lấy mẫu G trajectory (mỗi cái là một chuỗi hành động hoàn
    chỉnh) cho CÙNG một bài toán từ π_old. G ≠ số bước trong một trajectory."""
    return [generate_trajectory(policy, numbers, target, config) for _ in range(config.group_size)]


# ---------------------------------------------------------------------------
# 5. GROUP-RELATIVE ADVANTAGE
# ---------------------------------------------------------------------------

def calculate_group_advantages(rewards: Sequence[float], epsilon: float = 1e-4) -> List[float]:
    """
    A_i = (R_i - mean(R)) / (std(R) + epsilon) trong group.
    std == 0 (mọi trajectory cùng reward — ví dụ cùng thất bại y hệt): trả toàn 0,
    batch đó KHÔNG đóng góp cập nhật (ghi log zero_adv_groups), KHÔNG bịa advantage giả.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0] * n
    return [(r - mean) / (std + epsilon) for r in rewards]


# ---------------------------------------------------------------------------
# 6. GRPO OBJECTIVE
# ---------------------------------------------------------------------------

def calculate_grpo_loss(
    policy: PolicyNetwork,
    reference: Optional[PolicyNetwork],
    trajectories: Sequence[Trajectory],
    advantages: Sequence[float],
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Loss GRPO action-level trên một group.

    Trajectory nhiều bước: mỗi bước t là một "token", cùng chịu advantage cấp
    trajectory A_i (episode-level, group-relative); đóng góp của trajectory chia
    chuẩn hóa 1/|τ_i| (như paper GRPO). Tổng hợp:
        L = mean_i [ -1/|τ_i| Σ_t min(ρ_t·A_i, clip(ρ_t,1±ε)·A_i) ] + β·mean KL
    với ρ_t = exp(logπ_θ(a_t|s_t) - old_log_prob_t). old_log_prob lưu lúc sinh
    (= π_old); π_θ tính lại GIỮ graph để backprop; π_ref no_grad, chỉ dùng KL β>0.
    Output: (loss vô hướng có graph, stats: mean_ratio, clip_frac, kl, entropy).
    """
    policy.train()
    device = next(policy.parameters()).device
    total = torch.zeros((), device=device)
    used = 0
    ratio_logs: List[float] = []
    clip_count = 0
    kl_terms: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []

    for traj, adv in zip(trajectories, advantages):
        if adv == 0.0 or not traj.steps:
            continue
        traj_loss = torch.zeros((), device=device)
        for step in traj.steps:
            logits = policy(step.state, step.target, step.actions)
            log_probs = F.log_softmax(logits, dim=0)
            new_log_prob = log_probs[step.taken_idx]
            ratio = torch.exp(new_log_prob - torch.tensor(step.old_log_prob, device=device))
            weighted = ratio * adv / len(traj.steps)
            clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv / len(traj.steps)
            traj_loss = traj_loss - torch.min(weighted, clipped)
            ratio_logs.append(float(ratio.detach()))
            if bool(clipped < weighted):     # clip thực sự có tác dụng tại bước này
                clip_count += 1
            p = log_probs.exp()
            entropies.append(-(p * log_probs).sum())
            if kl_coefficient > 0.0 and reference is not None:
                with torch.no_grad():
                    ref_lp = F.log_softmax(reference(step.state, step.target, step.actions), dim=0)
                kl_terms.append((p * (log_probs - ref_lp)).sum())
        total = total + traj_loss
        used += 1

    if used == 0:
        loss = torch.zeros((), device=device, requires_grad=True)
    else:
        loss = total / used
        if kl_coefficient > 0.0 and kl_terms:
            loss = loss + kl_coefficient * torch.stack(kl_terms).mean()

    stats = {
        "mean_ratio": sum(ratio_logs) / len(ratio_logs) if ratio_logs else 1.0,
        "clip_frac": clip_count / len(ratio_logs) if ratio_logs else 0.0,
        "kl": float(torch.stack(kl_terms).mean().detach()) if kl_terms else 0.0,
        "entropy": float(torch.stack([e.detach() for e in entropies]).mean()) if entropies else 0.0,
    }
    return loss, stats


# ---------------------------------------------------------------------------
# 7. GRPO AGENT: train loop / evaluation / checkpoint / solve adapter
# ---------------------------------------------------------------------------

def make_data_splits(
    seed: Optional[int] = None,
    train_count: int = 60,
    val_count: int = 15,
    test_count: int = 15,
    min_numbers: int = 4,
    max_numbers: int = 6,
) -> Dict[str, List[Tuple[List[int], int]]]:
    """
    Sinh puzzle CÓ NGHIỆM (dùng lại generate_solvable_instances của rl_agent) và
    chia train / validation / test KHÔNG giao nhau — đánh giá không đo trên đúng
    puzzle đã huấn luyện. Trả về dict {'train': [...], 'val': [...], 'test': [...]}.
    """
    rng = random.Random(seed)
    train = generate_solvable_instances(train_count, rng=rng, min_numbers=min_numbers, max_numbers=max_numbers)
    val = generate_solvable_instances(val_count, rng=rng, min_numbers=min_numbers, max_numbers=max_numbers, exclude=train)
    test = generate_solvable_instances(test_count, rng=rng, min_numbers=min_numbers, max_numbers=max_numbers,
                                       exclude=train + val)
    return {"train": train, "val": val, "test": test}


class GRPOAgent:
    """
    Agent GRPO hoàn chỉnh cho Countdown: policy π_θ, reference π_ref (snapshot bất
    động, dùng khi kl_coefficient > 0), optimizer Adam, vòng huấn luyện group
    relative, đánh giá và checkpoint riêng (KHÔNG đụng vào q_table.json).
    """

    def __init__(self, config: Optional[GRPOConfig] = None, checkpoint_path: Optional[str] = None):
        self.config = config or GRPOConfig()
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            random.seed(self.config.seed)
        self.policy = PolicyNetwork(d_model=self.config.d_model).to(self.config.torch_device)
        self.reference = copy.deepcopy(self.policy).to(self.config.torch_device)
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.train_instances: List[Tuple[List[int], int]] = []
        # Toàn bộ split dữ liệu (train/val/test) được lưu kèm checkpoint để chế độ
        # eval đánh giá trên ĐÚNG tập test held-out, không trùng tập train.
        self.data_splits: Dict[str, List[Tuple[List[int], int]]] = {"train": [], "val": [], "test": []}
        self.policy_updates = 0
        self.history: List[Dict[str, Any]] = []

    # ---------------- Huấn luyện ----------------

    def train_grpo(
        self,
        instances: Sequence[Tuple[List[int], int]],
        val_instances: Optional[Sequence[Tuple[List[int], int]]] = None,
        checkpoint_path: Optional[str] = None,
        log_fn=lambda msg: print(msg),
    ) -> Dict[str, Any]:
        """
        Training loop GRPO:
          mỗi epoch: duyệt batch puzzle → generate_group (G trajectory) → verifier
          reward → group advantage → inner_epochs cập nhật gradient đã clip.
        Log mỗi epoch: mean reward, exact solve rate (val), valid rate, MAE,
        độ dài trajectory trung bình, entropy, loss, số lần cập nhật, tỉ lệ
        trajectory trùng lặp, số nhóm advantage 0 (std=0).
        """
        cfg = self.config
        self.train_instances = list(instances)
        rng = random.Random(cfg.seed)
        epoch_logs: List[Dict[str, Any]] = []
        steps_per_epoch = max(1, math.ceil(len(instances) / cfg.batch_size))

        for epoch in range(1, cfg.num_epochs + 1):
            order = list(range(len(instances)))
            rng.shuffle(order)
            rewards_log: List[float] = []
            lengths_log: List[int] = []
            losses_log: List[float] = []
            entropies_log: List[float] = []
            dup_log: List[float] = []
            zero_adv_groups = 0
            solved_groups = 0
            updates_this_epoch = 0

            for b in range(steps_per_epoch):
                batch_idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                if not batch_idx:
                    break
                groups: List[List[Trajectory]] = []
                advantages: List[List[float]] = []
                for idx in batch_idx:
                    numbers, target = instances[idx]
                    group = generate_group(self.policy, numbers, target, cfg)
                    advs = calculate_group_advantages([t.reward for t in group], cfg.adv_epsilon)
                    groups.append(group)
                    advantages.append(advs)
                    rewards_log.extend(t.reward for t in group)
                    lengths_log.extend(len(t.steps) for t in group)
                    keys = {t.key for t in group}
                    dup_log.append(1.0 - len(keys) / len(group))
                    if all(a == 0.0 for a in advs):
                        zero_adv_groups += 1
                    if any(t.success for t in group):
                        solved_groups += 1

                for _ in range(cfg.inner_epochs):
                    self.optimizer.zero_grad()
                    batch_loss = None
                    stats_accum: List[Dict[str, float]] = []
                    for group, advs in zip(groups, advantages):
                        loss, stats = calculate_grpo_loss(
                            self.policy, self.reference, group, advs,
                            cfg.clip_epsilon, cfg.kl_coefficient,
                        )
                        entropies_log.append(stats["entropy"])
                        if bool(loss.requires_grad) and float(loss.detach()) != 0.0:
                            batch_loss = loss if batch_loss is None else batch_loss + loss
                    if batch_loss is not None:
                        batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_gradient_norm)
                        self.optimizer.step()
                        self.policy_updates += 1
                        updates_this_epoch += 1
                        losses_log.append(float(batch_loss.detach()) / max(1, len(groups)))

            log: Dict[str, Any] = {
                "epoch": epoch,
                "mean_reward": sum(rewards_log) / len(rewards_log) if rewards_log else 0.0,
                "train_group_solve_rate": solved_groups / steps_per_epoch,
                "duplicate_rate": sum(dup_log) / len(dup_log) if dup_log else 0.0,
                "mean_traj_length": sum(lengths_log) / len(lengths_log) if lengths_log else 0.0,
                "loss": sum(losses_log) / len(losses_log) if losses_log else 0.0,
                "entropy": sum(entropies_log) / len(entropies_log) if entropies_log else 0.0,
                "policy_updates": self.policy_updates,
                "zero_adv_groups": zero_adv_groups,
            }
            if val_instances and epoch % cfg.evaluation_interval == 0:
                val_metrics = self.evaluate_grpo(list(val_instances), greedy=True)
                log.update({f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, list)})
            epoch_logs.append(log)
            self.history.append(log)
            if log_fn:
                log_fn(f"[GRPO epoch {epoch}/{cfg.num_epochs}] reward={log['mean_reward']:+.3f} "
                       f"train_solve={log['train_group_solve_rate']:.0%} loss={log['loss']:+.4f} "
                       f"entropy={log['entropy']:.2f} updates={self.policy_updates} "
                       f"dup={log['duplicate_rate']:.0%}"
                       + (f" val_exact={log.get('val_exact_solve_rate', float('nan')):.0%}"
                          if "val_exact_solve_rate" in log else ""))
            if checkpoint_path and epoch % cfg.checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_path)

        if checkpoint_path:
            self.save_checkpoint(checkpoint_path)
        return {"epochs": epoch_logs, "policy_updates": self.policy_updates}

    # ---------------- Đánh giá ----------------

    def evaluate_grpo(
        self,
        instances: Sequence[Tuple[List[int], int]],
        greedy: bool = True,
        best_of_n: int = 1,
    ) -> Dict[str, Any]:
        """
        Đánh giá policy trên tập puzzle (held-out). greedy=True: rollout argmax
        (chính sách thuần). best_of_n>1: lấy mẫu N trajectory rồi chọn nghiệm tốt
        nhất ĐÃ ĐƯỢC VERIFIER xác nhận (inference-time selection, có ghi rõ).
        Metrics: exact_solve_rate, valid_solution_rate, mean_abs_error,
        mean_traj_length, mean env reward.
        """
        exact = valid_count = 0
        errors: List[int] = []
        lengths: List[int] = []
        for numbers, target in instances:
            trajs: List[Trajectory] = []
            for _ in range(max(1, best_of_n)):
                trajs.append(generate_trajectory(self.policy, numbers, target, self.config, greedy=greedy))
            verified_success = [t for t in trajs if t.success]
            chosen = verified_success[0] if verified_success else max(trajs, key=lambda t: (t.valid, t.reward))
            if chosen.valid:
                valid_count += 1
            if verified_success:
                exact += 1
                errors.append(0)
            elif chosen.min_error is not None:
                errors.append(chosen.min_error)
            lengths.append(len(chosen.steps))
        n = len(instances) or 1
        return {
            "instances": len(instances),
            "exact_solve_rate": exact / n,
            "valid_solution_rate": valid_count / n,
            "mean_abs_error": sum(errors) / len(errors) if errors else 0.0,
            "mean_traj_length": sum(lengths) / n,
        }

    # ---------------- Inference adapter cho chế độ solve ----------------

    def solve(
        self,
        numbers: Sequence[int],
        target: int,
        recorder=None,
        fallback_group_size: int = 32,
    ) -> Tuple[Optional[Tuple[ActionKey, ...]], Dict[str, Any]]:
        """
        Giải MỘT bài toán bằng policy đã huấn luyện (chế độ solve của main.py).
        Chiến lược: (1) rollout greedy thuần; (2) nếu chưa đạt, best-of-N từ
        policy sample. Nghiệm chỉ được trả về khi VERIFIER ĐỘC LẬP xác nhận —
        không bao giờ trả "lời giải" chưa kiểm chứng.

        Trả về (operations, stats) với operations là tuple 4-tuple đúng chuẩn
        (left, op, right, result) để build_steps/save_results dùng lại không đổi;
        None nếu không có nghiệm được xác nhận (giống hợp đồng MCTS cũ).
        """
        cfg = self.config
        trajectories: List[Trajectory] = []
        greedy_traj = generate_trajectory(self.policy, numbers, target, cfg, greedy=True)
        trajectories.append(greedy_traj)
        if not greedy_traj.success:
            trajectories.extend(generate_group(self.policy, numbers, target,
                                                _group_config(cfg, fallback_group_size)))
        solved = [t for t in trajectories if t.success]
        best = solved[0] if solved else None

        if recorder is not None:
            self._record(trajectories[0] if best is None else best, numbers, target, recorder)

        stats = {
            "algorithm": "GRPO (Group Relative Policy Optimization)",
            "trajectories_generated": len(trajectories),
            "verified": bool(solved),
            "policy_updates_at_load": self.policy_updates,
            "solution_depth": len(best.actions) if best else 0,
        }
        if best is None:
            return None, stats
        return tuple(best.actions), stats

    @staticmethod
    def _record(traj: Trajectory, numbers: Sequence[int], target: int, recorder) -> None:
        """Ghi trajectory GRPO vào SearchTreeRecorder hiện có (tầng 0..2) — dùng
        lại API chung start/add_transition/finish, không sửa tree_visualizer.py.
        next_state được suy ra bằng cách tra đúng hành động 5-phần-tử trong
        get_valid_actions (CountdownEnv.step_state tin vào next_state dựng sẵn
        trong action, nên KHÔNG được truyền None)."""
        env = CountdownEnv(list(numbers), target)
        state = env.reset()
        root_id = recorder.start(state, visits=1, q_value=traj.env_reward)
        parent_id = root_id
        for depth, (left, op, right, res) in enumerate(traj.actions, start=1):
            match = next((a for a in env.get_valid_actions(state)
                          if (a[0], a[1], a[2], a[3]) == (left, op, right, res)), None)
            if match is None:
                break
            next_state = match[4]
            if depth <= recorder.MAX_DEPTH:
                node_id = recorder.add_transition(parent_id, next_state, depth,
                                                  (left, op, right, res), visits=1, q_value=traj.env_reward)
                if node_id is not None:
                    parent_id = node_id
            state = next_state
        recorder.finish(traj.success, visits=1, q_value=traj.reward)

    # ---------------- Checkpoint ----------------
    #
    # Checkpoint GRPO là MỘT THƯ MỤC (mặc định grpo_checkpoint/) chứa:
    #   policy.pt / reference.pt / optimizer.pt : tensor state_dict, nạp bằng
    #       torch.load(weights_only=True) — an toàn, không thực thi pickle tuple
    #       object tùy ý (tránh rủi ro deserialization code-execution).
    #   meta.json : config, danh sách instance train, số cập nhật, lịch sử.
    # Hoàn toàn TÁCH BIỆT với q_table.json của Q-Learning.

    def save_checkpoint(self, path) -> None:
        """Lưu checkpoint GRPO riêng ra thư mục (policy + reference + optimizer + meta)."""
        ckpt_dir = Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.policy.state_dict().items()}, ckpt_dir / "policy.pt")
        torch.save({k: v.cpu() for k, v in self.reference.state_dict().items()}, ckpt_dir / "reference.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        meta = {
            "version": 1,
            "config": {k: (str(v) if k == "device" else v) for k, v in self.config.__dict__.items()},
            "train_instances": [[list(n), t] for n, t in self.train_instances],
            "data_splits": {k: [[list(n), t] for n, t in v] for k, v in self.data_splits.items()},
            "policy_updates": self.policy_updates,
            "history": self.history,
        }
        (ckpt_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path) -> "GRPOAgent":
        """Nạp checkpoint GRPO từ thư mục, dựng lại agent (π_θ + π_ref + optimizer + config).
        File không tồn tại → FileNotFoundError (CLI sẽ thông báo thân thiện)."""
        ckpt_dir = Path(path)
        policy_file = ckpt_dir / "policy.pt"
        meta_file = ckpt_dir / "meta.json"
        if not policy_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Checkpoint GRPO không hợp lệ tại {ckpt_dir} (thiếu policy.pt/meta.json).")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        cfg_kwargs = dict(meta["config"])
        device = cfg_kwargs.pop("device", "cpu")
        cfg_kwargs.pop("torch_device", None)
        cfg = GRPOConfig(device=device, **cfg_kwargs)
        agent = cls(config=cfg)
        agent.policy.load_state_dict(torch.load(policy_file, map_location="cpu", weights_only=True))
        reference_file = ckpt_dir / "reference.pt"
        if reference_file.exists():
            agent.reference.load_state_dict(torch.load(reference_file, map_location="cpu", weights_only=True))
        optimizer_file = ckpt_dir / "optimizer.pt"
        if optimizer_file.exists():
            try:
                agent.optimizer.load_state_dict(
                    torch.load(optimizer_file, map_location="cpu", weights_only=True))
            except (ValueError, RuntimeError):
                pass  # optimizer reset vẫn an toàn khi chỉ inference
        agent.train_instances = [([int(x) for x in n], int(t)) for n, t in meta.get("train_instances", [])]
        agent.data_splits = {
            k: [([int(x) for x in n], int(t)) for n, t in v]
            for k, v in meta.get("data_splits", {"train": [], "val": [], "test": []}).items()
        }
        agent.policy_updates = int(meta.get("policy_updates", 0))
        agent.history = meta.get("history", [])
        return agent


def _group_config(cfg: GRPOConfig, group_size: int) -> GRPOConfig:
    """Bản sao config với group_size khác (dùng cho best-of-N lúc inference)."""
    new = copy.copy(cfg)
    new.group_size = group_size
    return new


# ---------------------------------------------------------------------------
# 8. GIAO DIỆN TƯƠNG THÍCH solve_countdown_rl (adapter, không sửa rl_agent.py)
# ---------------------------------------------------------------------------

def solve_countdown_grpo(
    numbers: List[int],
    target: int,
    recorder=None,
    require_use_all: bool = True,
    checkpoint: Optional[str] = None,
    agent: Optional[GRPOAgent] = None,
    **kwargs,
) -> Tuple[Optional[Tuple[ActionKey, ...]], Dict[str, Any]]:
    """
    Hàm giao diện GRPO song song với solve_countdown_rl của rl_agent:
    trả về (operations, stats) với operations là tuple 4-tuple chuẩn → build_steps
    / save_results / GUI của main.py dùng lại tự do, kiểu dữ liệu không đổi.
    Nếu chưa truyền agent, nạp từ `checkpoint` (mặc định grpo_checkpoint.pt).
    """
    if agent is None:
        path = checkpoint or str(Path(__file__).with_name("grpo_checkpoint.pt"))
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint GRPO tại {path}. "
                "Hãy chạy: python main.py --algo grpo --mode train trước."
            )
        agent = GRPOAgent.load_checkpoint(path)
    agent.config.require_use_all = require_use_all
    return agent.solve(list(numbers), int(target), recorder=recorder, **kwargs)

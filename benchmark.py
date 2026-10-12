"""
benchmark.py — So sánh trung thực các solver Countdown trên CÙNG một bộ puzzle.

Chạy:  py benchmark.py [--instances 20] [--seed 999] [--best-of-n 16]

So sánh:
  1. MCTS + PUCT (rl_agent.MCTSRLSolver) — tìm kiếm quy hoạch model-based.
  2. Q-Learning (rl_agent.QLearningAgent, Q-table đã lưu q_table.json) — greedy thuần.
  3. GRPO TRƯỚC huấn luyện (PolicyNetwork khởi tạo ngẫu nhiên, seed cố định).
  4. GRPO SAU HUẤN LUYỆN (checkpoint grpo_checkpoint/ do --algo grpo --mode train).
  5. Solver cổ điển: DFS quay lui exhaustive có memo (điểm tham chiếu "đúng nhất").

Đo trên mỗi solver: exact solve rate (nghiệm đạt target, đã verifier độc lập xác
nhận), valid solution rate, thời gian trung bình, số bước trung bình, sai lệch
trung bình so với target. Không khẳng định GRPO tốt hơn — chỉ in số liệu thực đo.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# Đảm bảo in tiếng Việt trên Windows console (cp1252) không lỗi — giống main.py.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from countdown_env import CountdownEnv
from grpo_agent import (
    GRPOAgent,
    GRPOConfig,
    generate_trajectory,
    solve_countdown_grpo,
    verify_trajectory,
)
from rl_agent import QLearningAgent, generate_solvable_instances, solve_countdown_rl

REQUIRE_USE_ALL = True


def classic_dfs(numbers: Tuple[int, ...], target: int) -> Optional[List[Tuple[int, str, int, int]]]:
    """
    Solver cổ điển: quay lui exhaustive trên MDP Countdown (mọi trạng thái được
    memo bằng trạng thái đã biết 'bất tử'). Không heuristic — đại diện cho tìm
    kiếm đầy đủ trong không gian hữu hạn. Trả về dãy 4-tuple hoặc None.
    """
    env = CountdownEnv(list(numbers), target, require_use_all=REQUIRE_USE_ALL)
    dead: set = set()

    def rec(state: Tuple[int, ...], path: List) -> bool:
        if state in dead:
            return False
        if len(state) == 1:
            return state[0] == target
        for action in env.get_valid_actions(state):
            key = (action[0], action[1], action[2], action[3])
            nxt = action[4]
            path.append(key)
            if rec(nxt, path):
                return True
            path.pop()
        dead.add(state)
        return False

    found = rec(tuple(sorted(numbers)), path := [])
    return path if found else None


def run_classic(instances: List[Tuple[List[int], int]]) -> Dict[str, float]:
    times, steps, exact, valid_sum, errors = [], [], 0, 0, []
    for numbers, target in instances:
        start = time.perf_counter()
        ops = classic_dfs(tuple(sorted(numbers)), target)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if ops is not None:
            v = verify_trajectory(numbers, target, ops, REQUIRE_USE_ALL)
            exact += int(v.success)
            valid_sum += int(v.valid)
            steps.append(len(ops))
            errors.append(0 if v.success else (v.min_error or 0))
        else:
            errors.append(target)  # không có nghiệm
    return summarize("Classic DFS (memo)", exact, valid_sum, len(instances), times, steps, errors)


def run_mcts(instances: List[Tuple[List[int], int]]) -> Dict[str, float]:
    times, steps, exact, valid_sum, errors = [], [], 0, 0, []
    for numbers, target in instances:
        start = time.perf_counter()
        ops, _ = solve_countdown_rl(list(numbers), target, algorithm="mcts",
                                    require_use_all=REQUIRE_USE_ALL)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if ops is not None:
            v = verify_trajectory(numbers, target, ops, REQUIRE_USE_ALL)
            exact += int(v.success)
            valid_sum += int(v.valid)
            steps.append(len(ops))
            errors.append(0 if v.success else (v.min_error or 0))
        else:
            errors.append(target)
    return summarize("MCTS + PUCT", exact, valid_sum, len(instances), times, steps, errors)


def run_qlearning(instances: List[Tuple[List[int], int]], qfile: str) -> Dict[str, float]:
    """Greedy THUẦN trên Q-table đã lưu (logic đồng bộ QLearningAgent.greedy_success
    nhưng ghi lại hành động để verifier kiểm chứng)."""
    agent = QLearningAgent(require_use_all=REQUIRE_USE_ALL, q_table_path=qfile)
    rng = random.Random(123)
    times, steps, exact, valid_sum, errors = [], [], 0, 0, []
    for numbers, target in instances:
        start = time.perf_counter()
        env = CountdownEnv(list(numbers), target, require_use_all=REQUIRE_USE_ALL)
        state = env.reset()
        chosen: List[Tuple[int, str, int, int]] = []
        while len(state) > 1:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break
            max_q = max(agent.get_q(state, target, a) for a in valid_actions)
            best = [a for a in valid_actions if agent.get_q(state, target, a) == max_q]
            action = rng.choice(best)
            chosen.append((action[0], action[1], action[2], action[3]))
            state = action[4]
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        v = verify_trajectory(numbers, target, chosen, REQUIRE_USE_ALL)
        exact += int(v.success)
        valid_sum += int(v.valid)
        steps.append(len(chosen))
        errors.append(0 if v.success else (v.min_error if v.min_error is not None else target))
    return summarize(f"Q-Learning (q_table.json, {len(agent.Q)} mục)", exact, valid_sum,
                     len(instances), times, steps, errors)


def run_grpo_untrained(instances: List[Tuple[List[int], int]], seed: int) -> Dict[str, float]:
    """GRPO TRƯỚC huấn luyện: policy khởi tạo ngẫu nhiên (chưa optimizer.step nào)."""
    cfg = GRPOConfig(seed=seed, require_use_all=REQUIRE_USE_ALL)
    agent = GRPOAgent(cfg)
    return evaluate_agent(agent, instances, label="GRPO trước huấn luyện (random init)")


def run_grpo_trained(instances: List[Tuple[List[int], int]], ckpt: str, best_of_n: int) -> List[Dict[str, float]]:
    """GRPO SAU huấn luyện: greedy thuần và best-of-N sampling (nếu checkpoint tồn tại)."""
    agent = GRPOAgent.load_checkpoint(ckpt)
    out = [evaluate_agent(agent, instances, label="GRPO sau huấn luyện (greedy thuần)", greedy=True)]
    out.append(evaluate_agent(agent, instances, label=f"GRPO sau huấn luyện (best-of-{best_of_n})",
                              greedy=False, best_of_n=best_of_n))
    return out


def evaluate_agent(agent: GRPOAgent, instances, label: str, greedy: bool = True,
                   best_of_n: int = 1) -> Dict[str, float]:
    times, steps, exact, valid_sum, errors = [], [], 0, 0, []
    for numbers, target in instances:
        start = time.perf_counter()
        ops, _ = agent.solve(numbers, target, recorder=None, fallback_group_size=best_of_n) \
            if best_of_n > 1 else (
                _greedy_ops(agent, numbers, target), None)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if ops is not None:
            exact += 1
            valid_sum += 1
            steps.append(len(ops))
            errors.append(0)
        else:
            traj = generate_trajectory(agent.policy, numbers, target, agent.config, greedy=greedy)
            valid_sum += int(traj.valid)
            steps.append(len(traj.actions))
            errors.append(traj.min_error if traj.min_error is not None else target)
    return summarize(label, exact, valid_sum, len(instances), times, steps, errors)


def _greedy_ops(agent: GRPOAgent, numbers, target):
    traj = generate_trajectory(agent.policy, numbers, target, agent.config, greedy=True)
    return traj.actions if traj.success else None


def summarize(label: str, exact: int, valid: int, total: int, times, steps, errors) -> Dict[str, float]:
    return {
        "solver": label,
        "exact_solve_rate": f"{exact}/{total} = {exact / max(1, total):.0%}",
        "valid_solution_rate": f"{valid}/{total} = {valid / max(1, total):.0%}",
        "mean_time_ms": f"{1000 * sum(times) / max(1, len(times)):.1f}",
        "mean_steps": f"{sum(steps) / max(1, len(steps)):.2f}",
        "mean_abs_error": f"{sum(errors) / max(1, len(errors)):.1f}",
    }


def print_table(rows: List[Dict[str, float]]) -> None:
    headers = ["solver", "exact_solve_rate", "valid_solution_rate", "mean_time_ms", "mean_steps", "mean_abs_error"]
    all_rows = rows + [dict(zip(headers, headers))]
    widths = [max(len(str(r[h])) for r in all_rows) for h in headers]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(w) for h, w in zip(headers, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MCTS / Q-Learning / GRPO / DFS cổ điển.")
    parser.add_argument("--instances", type=int, default=20, help="Số puzzle test (mặc định 20).")
    parser.add_argument("--seed", type=int, default=999, help="Seed sinh bộ puzzle (mặc định 999).")
    parser.add_argument("--best-of-n", type=int, default=16, help="N cho GRPO best-of-N (mặc định 16).")
    parser.add_argument("--qfile", default="q_table.json")
    parser.add_argument("--grpo-file", default="grpo_checkpoint")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    instances = generate_solvable_instances(args.instances, rng=rng)
    print(f"Bộ puzzle: {len(instances)} instance sinh ngẫu nhiên, seed={args.seed}, "
          f"luật require_use_all={REQUIRE_USE_ALL}, mọi nghiệm đều qua verifier độc lập.\n")

    rows: List[Dict[str, float]] = []
    rows.append(run_classic(instances))
    rows.append(run_mcts(instances))
    rows.append(run_qlearning(instances, args.qfile))
    rows.append(run_grpo_untrained(instances, seed=args.seed))
    import pathlib
    if pathlib.Path(args.grpo_file, "policy.pt").exists():
        rows.extend(run_grpo_trained(instances, args.grpo_file, args.best_of_n))
    else:
        print(f"(Bỏ qua GRPO đã train — không có checkpoint tại {args.grpo_file}/; "
              "chạy: python main.py --algo grpo --mode train)")
    print_table(rows)


if __name__ == "__main__":
    main()

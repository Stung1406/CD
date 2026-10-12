"""
test_grpo.py — Unit test cho module GRPO + hồi quy cho MCTS/Q-Learning.

Chạy:  py -m unittest test_grpo -v
Không cần GPU, không cần file dữ liệu ngoài; checkpoint ghi vào thư mục tạm.
"""

from __future__ import annotations

import copy
import math
import random
import tempfile
import unittest
from pathlib import Path

import torch

from countdown_env import CountdownEnv
from grpo_agent import (
    GRPOAgent,
    GRPOConfig,
    PolicyNetwork,
    calculate_grpo_loss,
    calculate_group_advantages,
    calculate_trajectory_reward,
    generate_group,
    generate_trajectory,
    make_data_splits,
    solve_countdown_grpo,
    verify_trajectory,
)
from rl_agent import MCTSRLSolver, QLearningAgent, solve_countdown_rl

NUMBERS = [3, 6, 7, 8, 25, 50]
TARGET = 952
# Lời giải kinh điển (đã kiểm chứng tay): dùng HẾT 6 số, mọi phép hợp lệ.
KNOWN_SOLUTION = [(25, "-", 6, 19), (3, "+", 7, 10), (19, "*", 50, 950), (10, "-", 8, 2), (2, "+", 950, 952)]


class TestEnvRegression(unittest.TestCase):
    """Hồi quy: CountdownEnv (State/Action/P/R) giữ nguyên hành vi cho mọi solver."""

    def test_transition_deterministic(self):
        env = CountdownEnv(NUMBERS, TARGET)
        actions = env.get_valid_actions()
        self.assertTrue(actions)
        a = actions[0]
        s1, r1, d1, ok1 = env.step_state(env.initial_numbers, a)
        s2, r2, d2, ok2 = env.step_state(env.initial_numbers, a)
        self.assertEqual((s1, r1, d1, ok1), (s2, r2, d2, ok2))  # P(s'|s,a) = 1

    def test_action_rules(self):
        env = CountdownEnv(NUMBERS, TARGET)
        for state in [env.initial_numbers, (2, 5, 5, 25)]:
            for left, op, right, res, nxt in env.get_valid_actions(state):
                self.assertGreater(res, 0)
                if op == "/":
                    self.assertNotEqual(right, 0)
                    self.assertEqual(left % right, 0)
                if op == "-":
                    self.assertEqual(left - right, res)
                    self.assertGreater(res, 0)
                self.assertEqual(len(nxt), len(state) - 1)

    def test_terminal_rewards(self):
        env = CountdownEnv(NUMBERS, TARGET)
        # terminal thành công: state cuối == target sau 1 bước chót
        s = (2, 950)
        a = next(a for a in env.get_valid_actions(s) if a[3] == TARGET)
        nxt, r, done, ok = env.step_state(s, a)
        self.assertTrue(done and ok)
        self.assertEqual(r, 100.0)
        nxt, r, done, ok = env.step_state((1, 950), (1, "+", 950, 951, (951,)))
        self.assertTrue(done and not ok)
        self.assertLess(r, 0.0)


class TestVerifier(unittest.TestCase):
    """Verifier độc lập: nhận đúng lời giải thật, từ chối lời giải bịa."""

    def test_accepts_real_solution(self):
        result = verify_trajectory(NUMBERS, TARGET, KNOWN_SOLUTION, require_use_all=True)
        self.assertTrue(result.valid and result.success)
        self.assertEqual(result.final_value, TARGET)
        self.assertEqual(result.reason, "success")

    def test_rejects_reused_number(self):
        bad = KNOWN_SOLUTION + [(25, "+", 19, 44)]  # 25 đã dùng hết
        result = verify_trajectory(NUMBERS, TARGET, bad, require_use_all=True)
        self.assertFalse(result.valid)

    def test_rejects_foreign_number(self):
        bad = [(25, "-", 6, 19), (3, "+", 7, 10), (19, "*", 50, 950), (10, "-", 8, 2), (2, "+", 951, 953)]
        result = verify_trajectory(NUMBERS, TARGET, bad, require_use_all=True)
        self.assertFalse(result.valid)

    def test_rejects_bad_arithmetic(self):
        bad = [(25, "-", 6, 20)] + KNOWN_SOLUTION[1:]
        result = verify_trajectory(NUMBERS, TARGET, bad, require_use_all=True)
        self.assertFalse(result.valid)

    def test_rejects_division_by_zero_and_nonexact(self):
        for bad in ([(5, "/", 0, 1)], [(7, "/", 3, 2)]):
            result = verify_trajectory([3, 5, 6, 7], 10, bad, require_use_all=True)
            self.assertFalse(result.valid)

    def test_rejects_nonpositive_subtraction(self):
        result = verify_trajectory([3, 5, 6, 7], 10, [(3, "-", 5, -2)], require_use_all=True)
        self.assertFalse(result.valid)

    def test_require_use_all_enforced(self):
        # Chỉ dùng 2 số đầu ra target nhưng còn số chưa dùng -> thất bại theo luật dự án
        result = verify_trajectory([25, 50, 3, 7], 25, [(50, "-", 25, 25)], require_use_all=True)
        self.assertTrue(result.valid)
        self.assertFalse(result.success)
        # Với luật partial (require_use_all=False) thì đạt target ngay
        result = verify_trajectory([25, 50, 3, 7], 25, [(50, "-", 25, 25)], require_use_all=False)
        self.assertTrue(result.success)

    def test_trajectory_reward_values(self):
        r, res = calculate_trajectory_reward(NUMBERS, TARGET, KNOWN_SOLUTION, True)
        self.assertEqual(r, 1.0)
        self.assertTrue(res.success)
        r_bad, res_bad = calculate_trajectory_reward(NUMBERS, TARGET, [(25, "-", 6, 99)], True)
        self.assertEqual(r_bad, -1.0)
        r_miss, _ = calculate_trajectory_reward([2, 3, 4, 5], 100, [(2, "+", 3, 5), (4, "*", 5, 20), (5, "+", 20, 25)], True)
        self.assertLess(r_miss, 0.0)
        self.assertGreater(r_miss, -1.0)


class TestPolicyNetwork(unittest.TestCase):
    """Policy π(a|s,target): logits biến thiên theo số hành động, có gradient thật."""

    def setUp(self):
        torch.manual_seed(0)
        self.policy = PolicyNetwork()
        self.env = CountdownEnv(NUMBERS, TARGET)
        self.actions = tuple((a[0], a[1], a[2], a[3]) for a in self.env.get_valid_actions())

    def test_logits_shape_matches_valid_actions(self):
        logits = self.policy(self.env.initial_numbers, TARGET, self.actions)
        self.assertEqual(tuple(logits.shape), (len(self.actions),))
        probs = torch.softmax(logits, dim=0)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_variable_action_count(self):
        few = self.policy((2, 950), TARGET, ((2, "+", 950, 952),))
        self.assertEqual(tuple(few.shape), (1,))

    def test_log_prob_differentiable_all_params(self):
        logits = self.policy(self.env.initial_numbers, TARGET, self.actions)
        log_prob = torch.log_softmax(logits, dim=0)[2]
        log_prob.backward()
        grads = [p.grad for p in self.policy.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))

    def test_sample_valid_and_log_prob_matches(self):
        cfg = GRPOConfig(seed=42)
        traj = generate_trajectory(self.policy, NUMBERS, TARGET, cfg)
        self.assertTrue(traj.steps)
        step = traj.steps[0]
        logits = self.policy(step.state, step.target, step.actions)
        lp = float(torch.log_softmax(logits, dim=0)[step.taken_idx])
        self.assertAlmostEqual(lp, step.old_log_prob, places=5)


class TestAdvantage(unittest.TestCase):
    def test_group_relative_values(self):
        advs = calculate_group_advantages([1.0, 0.0, -1.0, 0.0], epsilon=0.0)
        mean = 0.0
        std = math.sqrt(sum(r ** 2 for r in [1.0, 0.0, -1.0, 0.0]) / 4)
        expected = [(r - mean) / std for r in [1.0, 0.0, -1.0, 0.0]]
        for a, e in zip(advs, expected):
            self.assertAlmostEqual(a, e, places=5)

    def test_zero_std_returns_zeros(self):
        self.assertEqual(calculate_group_advantages([0.5, 0.5, 0.5]), [0.0, 0.0, 0.0])

    def test_empty(self):
        self.assertEqual(calculate_group_advantages([]), [])


class TestGRPOLoss(unittest.TestCase):
    """Objective GRPO: hữu hạn, có gradient, clip hoạt động, old-vs-current nhất quán."""

    def setUp(self):
        torch.manual_seed(1)
        self.cfg = GRPOConfig(seed=1, group_size=3)
        self.agent = GRPOAgent(self.cfg)
        self.group = generate_group(self.agent.policy, NUMBERS, TARGET, self.cfg)
        self.advs = calculate_group_advantages([t.reward for t in self.group])

    def test_loss_finite_and_graphed(self):
        loss, stats = calculate_grpo_loss(self.agent.policy, self.agent.reference, self.group, self.advs)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertTrue(loss.requires_grad)

    def test_ratio_is_one_when_policy_unchanged(self):
        _, stats = calculate_grpo_loss(self.agent.policy, None, self.group, self.advs)
        if stats["mean_ratio"] != 1.0:  # batch toàn adv=0 -> loss 0, ratio mặc định
            self.assertTrue(any(a != 0 for a in self.advs))
            self.assertAlmostEqual(stats["mean_ratio"], 1.0, places=4)

    def test_gradient_flows_and_params_update(self):
        params_before = [p.detach().clone() for p in self.agent.policy.parameters()]
        loss, _ = calculate_grpo_loss(self.agent.policy, None, self.group, self.advs)
        if all(a == 0 for a in self.advs):
            self.skipTest("nhóm đồng nhất, không có tín hiệu cập nhật (hành vi đúng)")
        self.agent.optimizer.zero_grad()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.agent.policy.parameters()))
        self.agent.optimizer.step()
        changed = any(not torch.allclose(p0, p1) for p0, p1 in
                      zip(params_before, self.agent.policy.parameters()))
        self.assertTrue(changed, "optimizer.step() phải thay đổi tham số π_θ")

    def test_policy_gradient_direction_increases_good_action(self):
        """Advantage dương làm log-prob hành động đã chọn TĂNG sau few-step update."""
        traj = generate_trajectory(self.agent.policy, NUMBERS, TARGET, GRPOConfig(seed=5))
        step = traj.steps[0]
        advs = [1.0]  # ép advantage dương (trường hợp trajectory tốt nhất nhóm)
        opt = torch.optim.SGD(self.agent.policy.parameters(), lr=0.05)
        for _ in range(5):
            loss, _ = calculate_grpo_loss(self.agent.policy, None, [traj], advs, clip_epsilon=10.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        logits = self.agent.policy(step.state, step.target, step.actions)
        new_lp = float(torch.log_softmax(logits, dim=0)[step.taken_idx])
        self.assertGreater(new_lp, step.old_log_prob)

    def test_clipping_limits_ratio(self):
        traj = generate_trajectory(self.agent.policy, NUMBERS, TARGET, GRPOConfig(seed=7))
        # bịa old_log_prob sai lệch lớn để ratio to -> clip phải có tác dụng
        traj.steps[0].old_log_prob -= 5.0
        _, stats = calculate_grpo_loss(self.agent.policy, None, [traj], [1.0], clip_epsilon=0.2)
        self.assertGreater(stats["clip_frac"], 0.0)

    def test_kl_term_active_with_reference(self):
        ref = copy.deepcopy(self.agent.policy)
        with torch.no_grad():
            for p in ref.parameters():
                p.add_(0.5)  # tách reference ra để KL > 0
        loss, stats = calculate_grpo_loss(self.agent.policy, ref, self.group, [1.0] * len(self.group),
                                          kl_coefficient=0.1)
        self.assertGreater(stats["kl"], 0.0)


class TestGroupGeneration(unittest.TestCase):
    def test_group_size_and_records(self):
        cfg = GRPOConfig(seed=3, group_size=5)
        agent = GRPOAgent(cfg)
        group = generate_group(agent.policy, NUMBERS, TARGET, cfg)
        self.assertEqual(len(group), 5)                       # 5 trajectory
        self.assertTrue(all(len(t.steps) >= 5 for t in group))  # mỗi cái nhiều bước (k-1=5 bước)
        for t in group:
            self.assertEqual(len(t.steps), len(t.actions))
            self.assertIsInstance(t.reward, float)
            self.assertTrue(math.isfinite(t.reward))

    def test_success_field_matches_verifier(self):
        cfg = GRPOConfig(seed=9)
        agent = GRPOAgent(cfg)
        t = generate_trajectory(agent.policy, NUMBERS, TARGET, cfg)
        v = verify_trajectory(NUMBERS, TARGET, t.actions, True)
        self.assertEqual(t.success, v.success)
        self.assertEqual(t.valid, v.valid)


class TestCheckpoint(unittest.TestCase):
    def test_save_load_reproduces_logits(self):
        cfg = GRPOConfig(seed=11)
        agent = GRPOAgent(cfg)
        env = CountdownEnv(NUMBERS, TARGET)
        actions = tuple((a[0], a[1], a[2], a[3]) for a in env.get_valid_actions())
        with torch.no_grad():
            before = agent.policy(env.initial_numbers, TARGET, actions)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grpo_ckpt"
            agent.save_checkpoint(path)
            self.assertTrue((path / "policy.pt").exists() and (path / "meta.json").exists())
            loaded = GRPOAgent.load_checkpoint(path)
            with torch.no_grad():
                after = loaded.policy(env.initial_numbers, TARGET, actions)
            self.assertTrue(torch.allclose(before, after, atol=1e-6))
            self.assertEqual(loaded.config.seed, 11)

    def test_checkpoint_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            GRPOAgent.load_checkpoint("khong_ton_tai/")


class TestTrainSmoke(unittest.TestCase):
    """train_grpo phải thực sự cập nhật tham số (không báo giả thành công)."""

    def test_train_updates_params(self):
        cfg = GRPOConfig(seed=2, num_epochs=1, batch_size=2, group_size=3,
                         learning_rate=1e-2, checkpoint_interval=1, evaluation_interval=1)
        agent = GRPOAgent(cfg)
        instances = [([2, 3, 7, 10], 24), ([4, 5, 6, 25], 120), ([3, 6, 7, 8], 63)]
        params_before = [p.detach().clone() for p in agent.policy.parameters()]
        with tempfile.TemporaryDirectory() as tmp:
            result = agent.train_grpo(instances, val_instances=instances[:1],
                                      checkpoint_path=Path(tmp) / "ckpt", log_fn=None)
            self.assertTrue((Path(tmp) / "ckpt" / "policy.pt").exists())
        self.assertGreater(agent.policy_updates, 0)
        self.assertTrue(any(not torch.allclose(p0, p1) for p0, p1 in
                            zip(params_before, agent.policy.parameters())))
        self.assertEqual(result["epochs"][0]["policy_updates"], agent.policy_updates)

    def test_data_splits_are_disjoint(self):
        splits = make_data_splits(seed=0, train_count=6, val_count=3, test_count=3)
        keys = {k: {(tuple(sorted(n)), t) for n, t in v} for k, v in splits.items()}
        self.assertEqual(len(keys["train"] & keys["val"]), 0)
        self.assertEqual(len(keys["train"] & keys["test"]), 0)


class TestSolveAdapter(unittest.TestCase):
    """solve GRPO trả đúng kiểu (tuple 4-tuple | None, stats) như MCTS cũ."""

    def test_solve_returns_contract_compatible(self):
        cfg = GRPOConfig(seed=4)
        agent = GRPOAgent(cfg)
        ops, stats = agent.solve(NUMBERS, TARGET)
        self.assertIsInstance(stats, dict)
        self.assertIn("algorithm", stats)
        if ops is not None:
            self.assertTrue(all(len(a) == 4 for a in ops))
            self.assertTrue(verify_trajectory(NUMBERS, TARGET, ops, True).success)

    def test_solve_countdown_grpo_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            solve_countdown_grpo(NUMBERS, TARGET, checkpoint="khong_co_day/")

    def test_solve_with_recorder_does_not_crash(self):
        """Hồi quy: solve qua đường CLI (có SearchTreeRecorder) phải chạy được
        _record mà không lỗi next_state=None (lỗi từng có ở phiên trước)."""
        from tree_visualizer import SearchTreeRecorder
        cfg = GRPOConfig(seed=4)
        agent = GRPOAgent(cfg)
        recorder = SearchTreeRecorder(TARGET, algorithm="RL (GRPO)")
        with tempfile.TemporaryDirectory() as tmp:
            dot = Path(tmp) / "t.dot"
            ops, stats = agent.solve(NUMBERS, TARGET, recorder=recorder)
            recorder.write_dot(dot)  # phải ghi được mà khôngexception
        self.assertIn("algorithm", stats)
        self.assertTrue(recorder.nodes)  # ít nhất có node gốc


class TestAlgoRegression(unittest.TestCase):
    """Hồi quy MCTS + Q-Learning: các thuật toán cũ vẫn chạy và vẫn đúng."""

    def test_mcts_still_solves(self):
        ops, stats = solve_countdown_rl(NUMBERS, TARGET, algorithm="mcts")
        self.assertIsNotNone(ops)
        self.assertIn("simulations", stats)
        result = verify_trajectory(NUMBERS, TARGET, ops, require_use_all=True)
        self.assertTrue(result.success, "MCTS phải còn ra nghiệm hợp lệ đã kiểm chứng")

    def test_qlearning_single_instance_still_solves(self):
        agent = QLearningAgent(numbers=[2, 3, 7, 10], target=24, max_episodes=4000,
                               alpha=0.3, gamma=0.95, epsilon=0.5)
        ops, stats = agent.train()
        self.assertIsNotNone(ops)
        self.assertIn("episodes_trained", stats)
        self.assertTrue(verify_trajectory([2, 3, 7, 10], 24, ops, require_use_all=True).success)

    def test_mcts_solution_feeds_build_steps(self):
        """Tương thích pipeline main.py: ops của MCTS đi qua build_steps hợp lệ."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from main import build_steps, read_problem
        numbers, target = read_problem(Path(__file__).parent / "input.txt")
        ops, _ = solve_countdown_rl(numbers, target, algorithm="mcts")
        steps, expression = build_steps(numbers, ops)
        self.assertEqual(len(steps), len(ops))
        self.assertIsNotNone(expression)


if __name__ == "__main__":
    unittest.main(verbosity=2)

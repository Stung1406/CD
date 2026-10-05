"""
rl_agent.py — Các thuật toán Học tăng cường (Reinforcement Learning) cho bài toán Countdown:

1. MCTS-RL (Monte Carlo Tree Search with PUCT & Q-Values):
   Mô hình Model-Based RL tương tự AlphaZero:
   - Policy Prior P(s, a): Ước tính xác suất tiên nghiệm dựa trên hàm tiềm năng khoảng cách
     và quan hệ ước số với Target.
   - Selection bằng công thức PUCT:
       PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
   - Expansion: Mở rộng các trạng thái con hợp lệ từ môi trường CountdownEnv.
   - Rollout: Mô phỏng chính sách guided rollout với chiết khấu gamma.
   - Backpropagation: Cập nhật visit counts N(s, a) và Q-values Q(s, a).
   - Multi-Restart: Tự động khởi động lại nếu cây bị rơi vào nhánh tắc nghẽn, đảm bảo hội tụ 100%.

2. Q-Learning Agent (Temporal Difference Learning):
   Mô hình Model-Free TD-Learning:
   - Học trực tiếp hàm giá trị hành động Q(s, a) qua các episode trải nghiệm.
   - Chiến lược epsilon-greedy thăm dò (exploration) kết hợp khai thác (exploitation).
   - Cập nhật Bellman Equation:
       Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
"""

import math
import random
import time
from collections import defaultdict
from typing import Tuple, List, Dict, Any, Optional

from countdown_env import CountdownEnv


class MCTSNode:
    """Nút trên cây tìm kiếm MCTS lưu trạng thái, xác suất tiên nghiệm P, số lượt ghé thăm N và giá trị Q."""

    def __init__(self, state: Tuple[int, ...], parent=None, action=None, prior: float = 1.0):
        self.state = state
        self.parent = parent
        self.action = action  # (left, op, right, res, next_state)
        self.prior = prior    # Policy prior P(s, a)
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions: Optional[List[Tuple[Any, float]]] = None
        self.is_terminal = False
        self.is_success = False

    @property
    def q_value(self) -> float:
        """Giá trị hành động trung bình Q(s, a)."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def puct(self, c_puct: float = 2.0) -> float:
        """Công thức PUCT (AlphaZero) cân bằng khai thác và thăm dò có định hướng."""
        if self.visits == 0:
            return float("inf")
        u = c_puct * self.prior * (math.sqrt(self.parent.visits) / (1 + self.visits))
        return self.q_value + u


class MCTSRLSolver:
    """
    Thuật toán Học tăng cường MCTS (Model-Based RL with PUCT).
    Lập kế hoạch tối ưu chuỗi phép tính Countdown bằng cây quyết định học tăng cường.
    """

    def __init__(
        self,
        numbers: List[int],
        target: int,
        require_use_all: bool = True,
        max_restarts: int = 10,
        max_sims_per_restart: int = 3000,
        c_puct: float = 2.0,
        recorder=None,
    ):
        self.env = CountdownEnv(numbers, target, require_use_all=require_use_all)
        self.numbers = numbers
        self.target = target
        self.require_use_all = require_use_all
        self.max_restarts = max_restarts
        self.max_sims_per_restart = max_sims_per_restart
        self.c_puct = c_puct
        self.recorder = recorder
        self.node_id_map: Dict[int, str] = {}
        self.solution_node: Optional[MCTSNode] = None

    def compute_prior(self, state: Tuple[int, ...], action: Tuple[int, str, int, int, Tuple[int, ...]]) -> float:
        """Tính điểm heuristic policy prior P(s, a) phản ánh chất lượng hành động."""
        res = action[3]
        dist = abs(res - self.target)
        factor_bonus = 5.0 if (res > 1 and self.target % res == 0) else 0.0
        score = - (dist / max(1, self.target)) + factor_bonus
        return score

    def _get_actions_with_priors(self, state: Tuple[int, ...]) -> List[Tuple[Any, float]]:
        actions = self.env.get_valid_actions(state)
        if not actions:
            return []
        priors = [self.compute_prior(state, a) for a in actions]
        max_p = max(priors) if priors else 0.0
        exp_p = [math.exp(p - max_p) for p in priors]
        sum_exp = sum(exp_p) if exp_p else 1.0
        probs = [p / sum_exp for p in exp_p]
        return list(zip(actions, probs))

    def search(self) -> Tuple[Optional[Tuple[Tuple[int, str, int, int], ...]], Dict[str, Any]]:
        """Thực hiện tìm kiếm MCTS với Multi-Restart để đảm bảo tìm ra nghiệm."""
        initial_state = self.env.reset()
        if len(initial_state) == 1 and initial_state[0] == self.target:
            return (), {"algorithm": "MCTS-RL (PUCT)", "simulations": 0, "root_visits": 1}

        total_simulations = 0
        best_root = None

        for restart in range(self.max_restarts):
            root = MCTSNode(initial_state, prior=1.0)
            best_root = root

            if self.recorder is not None and restart == 0:
                root_id = self.recorder.start(root.state, visits=1, q_value=0.0)
                self.node_id_map[id(root)] = root_id

            root.untried_actions = self._get_actions_with_priors(root.state)

            for sim in range(self.max_sims_per_restart):
                total_simulations += 1
                node = root
                depth = 0

                # 1. Selection
                while not node.untried_actions and node.children:
                    node = max(node.children, key=lambda ch: ch.puct(self.c_puct))
                    depth += 1

                # 2. Expansion
                if node.untried_actions:
                    # Chọn hành động có prior cao nhất
                    node.untried_actions.sort(key=lambda item: item[1])
                    action, prior = node.untried_actions.pop()

                    next_state, reward, done, is_success = self.env.step_state(node.state, action)
                    child = MCTSNode(next_state, parent=node, action=action, prior=prior)
                    child.is_terminal = done
                    child.is_success = is_success
                    node.children.append(child)

                    # Ghi nhận vào cây biểu diễn nếu ở tầng đầu
                    if self.recorder is not None and restart == 0 and depth < self.recorder.MAX_DEPTH:
                        parent_id = self.node_id_map.get(id(node))
                        if parent_id is not None:
                            op_info = (action[0], action[1], action[2], action[3])
                            child_id = self.recorder.add_transition(
                                parent_id, next_state, depth + 1, op_info, visits=1, q_value=reward
                            )
                            if child_id is not None:
                                self.node_id_map[id(child)] = child_id

                    if is_success:
                        self.solution_node = child
                        self.backpropagate(child, 100.0, record_tree=(restart == 0))
                        if self.recorder is not None:
                            self.recorder.finish(True, visits=root.visits, q_value=root.q_value)
                        ops = self.extract_path(child)
                        stats = {
                            "algorithm": "MCTS-RL (Monte Carlo Tree Search + PUCT)",
                            "simulations": total_simulations,
                            "restarts": restart,
                            "root_visits": root.visits,
                            "root_q_value": root.q_value,
                            "solution_depth": len(ops),
                        }
                        return ops, stats

                    child.untried_actions = self._get_actions_with_priors(next_state)

                    # 3. Rollout
                    val = self.rollout(child)

                    # 4. Backpropagation
                    self.backpropagate(child, val, record_tree=(restart == 0))
                else:
                    self.backpropagate(node, -1.0, record_tree=(restart == 0))

        if self.recorder is not None:
            visits = best_root.visits if best_root else total_simulations
            q_val = best_root.q_value if best_root else 0.0
            self.recorder.finish(False, visits=visits, q_value=q_val)

        stats = {
            "algorithm": "MCTS-RL (Monte Carlo Tree Search + PUCT)",
            "simulations": total_simulations,
            "restarts": self.max_restarts,
            "root_visits": best_root.visits if best_root else 0,
            "root_q_value": best_root.q_value if best_root else 0.0,
            "solution_depth": 0,
        }
        return None, stats

    def rollout(self, node: MCTSNode) -> float:
        """Mô phỏng chính sách guided rollout với chiết khấu gamma."""
        curr_state = node.state
        if len(curr_state) == 1:
            return 100.0 if curr_state[0] == self.target else -10.0

        gamma = 1.0
        tot = 0.0
        while len(curr_state) > 1:
            acts = self.env.get_valid_actions(curr_state)
            if not acts:
                break
            if len(curr_state) == 2 and self.require_use_all:
                hit = [a for a in acts if a[3] == self.target]
                if hit:
                    return 100.0 * gamma

            if random.random() < 0.7:
                best_a = min(acts, key=lambda a: abs(a[3] - self.target))
            else:
                best_a = random.choice(acts)

            curr_state, r, done, is_success = self.env.step_state(curr_state, best_a)
            tot += gamma * r
            gamma *= 0.95
            if is_success:
                return 100.0
            if done:
                break
        return tot

    def backpropagate(self, node: MCTSNode, value: float, record_tree: bool = True):
        """Cập nhật giá trị Q-value và số lần ghé thăm ngược về gốc."""
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.total_value += value
            if record_tree and self.recorder is not None and id(curr) in self.node_id_map:
                node_id = self.node_id_map[id(curr)]
                self.recorder.finish_node(
                    node_id,
                    curr.is_success or (self.solution_node is not None),
                    visits=curr.visits,
                    q_value=curr.q_value,
                )
            curr = curr.parent

    def extract_path(self, node: MCTSNode) -> Tuple[Tuple[int, str, int, int], ...]:
        """Trích xuất chuỗi các phép tính từ gốc tới nút nghiệm."""
        path = []
        curr = node
        while curr.parent is not None:
            a = curr.action
            path.append((a[0], a[1], a[2], a[3]))
            curr = curr.parent
        path.reverse()
        return tuple(path)


class QLearningAgent:
    """
    Thuật toán Học tăng cường Q-Learning (Model-Free TD-Learning).
    Học giá trị Q(s, a) qua các tập trải nghiệm (episodes) bằng phương trình Bellman.
    """

    def __init__(
        self,
        numbers: List[int],
        target: int,
        require_use_all: bool = True,
        alpha: float = 0.25,
        gamma: float = 0.95,
        epsilon: float = 0.45,
        max_episodes: int = 15000,
        recorder=None,
    ):
        self.env = CountdownEnv(numbers, target, require_use_all=require_use_all)
        self.numbers = numbers
        self.target = target
        self.require_use_all = require_use_all
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episodes = max_episodes
        self.recorder = recorder
        self.Q: Dict[Tuple[Tuple[int, ...], Tuple[int, str, int, int]], float] = defaultdict(float)

    @staticmethod
    def action_key(a: Tuple[int, str, int, int, Tuple[int, ...]]) -> Tuple[int, str, int, int]:
        return (a[0], a[1], a[2], a[3])

    def get_q(self, state: Tuple[int, ...], action: Tuple[int, str, int, int, Tuple[int, ...]]) -> float:
        return self.Q[(state, self.action_key(action))]

    def choose_action(
        self, state: Tuple[int, ...], valid_actions: List[Any], eps: float
    ) -> Optional[Any]:
        if not valid_actions:
            return None

        if len(state) == 2 and self.require_use_all:
            win_actions = [a for a in valid_actions if a[3] == self.target]
            if win_actions:
                return win_actions[0]

        if random.random() < eps:
            if random.random() < 0.6:
                return min(valid_actions, key=lambda a: abs(a[3] - self.target))
            return random.choice(valid_actions)
        else:
            max_q = max(self.get_q(state, a) for a in valid_actions)
            best_actions = [a for a in valid_actions if math.isclose(self.get_q(state, a), max_q, abs_tol=1e-6)]
            return random.choice(best_actions)

    def train(self) -> Tuple[Optional[Tuple[Tuple[int, str, int, int], ...]], Dict[str, Any]]:
        """Huấn luyện Agent qua nhiều episode cho đến khi tìm thấy nghiệm."""
        best_solution = None
        eps = self.epsilon
        decay = 0.9997

        root_state = self.env.reset()
        root_id = None
        if self.recorder is not None:
            root_id = self.recorder.start(root_state, visits=1, q_value=0.0)

        for ep in range(self.max_episodes):
            state = self.env.reset()
            trajectory = []

            while len(state) > 1:
                valid_actions = self.env.get_valid_actions(state)
                if not valid_actions:
                    break
                action = self.choose_action(state, valid_actions, eps)
                next_state, reward, done, is_success = self.env.step(action)
                trajectory.append((state, action, reward, next_state, done))

                if is_success:
                    best_solution = tuple(self.action_key(a) for _, a, _, _, _ in trajectory)
                    G = 100.0
                    for s, a, r, ns, d in reversed(trajectory):
                        ak = self.action_key(a)
                        self.Q[(s, ak)] += self.alpha * (G - self.Q[(s, ak)])
                        G *= self.gamma

                    if self.recorder is not None:
                        curr_parent = root_id
                        curr_depth = 1
                        for s, a, r, ns, d in trajectory:
                            if curr_depth <= self.recorder.MAX_DEPTH:
                                ak = self.action_key(a)
                                node_id = self.recorder.add_transition(
                                    curr_parent, ns, curr_depth, ak, visits=ep + 1, q_value=self.Q[(s, ak)]
                                )
                                self.recorder.finish_node(node_id, True, visits=ep + 1, q_value=self.Q[(s, ak)])
                                curr_parent = node_id
                                curr_depth += 1
                        self.recorder.finish(True, visits=ep + 1, q_value=100.0)

                    stats = {
                        "algorithm": "Q-Learning (Temporal Difference Learning)",
                        "episodes_trained": ep + 1,
                        "q_table_entries": len(self.Q),
                        "solution_depth": len(best_solution),
                    }
                    return best_solution, stats

                state = next_state
                if done:
                    break

            for s, a, r, ns, d in reversed(trajectory):
                ak = self.action_key(a)
                next_valid = self.env.get_valid_actions(ns)
                max_next_q = max([self.get_q(ns, na) for na in next_valid]) if next_valid else 0.0
                target_val = r if d else r + self.gamma * max_next_q
                self.Q[(s, ak)] += self.alpha * (target_val - self.Q[(s, ak)])

            eps = max(0.1, eps * decay)

        if self.recorder is not None:
            self.recorder.finish(False, visits=self.max_episodes, q_value=0.0)

        stats = {
            "algorithm": "Q-Learning (Temporal Difference Learning)",
            "episodes_trained": self.max_episodes,
            "q_table_entries": len(self.Q),
            "solution_depth": 0,
        }
        return None, stats


def solve_countdown_rl(
    numbers: List[int],
    target: int,
    recorder=None,
    algorithm: str = "mcts",
    require_use_all: bool = True,
    **kwargs,
) -> Tuple[Optional[Tuple[Tuple[int, str, int, int], ...]], Dict[str, Any]]:
    """
    Hàm giao diện chung giải quyết Countdown bằng Học tăng cường (RL).
    - algorithm: 'mcts' (mặc định - Model-based RL MCTS) hoặc 'qlearning' (Model-free TD learning).
    """
    if algorithm.lower() == "qlearning":
        agent = QLearningAgent(
            numbers=numbers,
            target=target,
            require_use_all=require_use_all,
            recorder=recorder,
            **kwargs,
        )
        return agent.train()
    else:
        solver = MCTSRLSolver(
            numbers=numbers,
            target=target,
            require_use_all=require_use_all,
            recorder=recorder,
            **kwargs,
        )
        return solver.search()

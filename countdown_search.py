"""Deterministic Countdown search, kept separate from the learned GRPO policy."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from countdown_env import CountdownEnv
from grpo_agent import ActionKey, verify_trajectory


def solve_countdown_exact(
    numbers: Sequence[int], target: int, require_use_all: bool = True
) -> Tuple[Optional[Tuple[ActionKey, ...]], Dict[str, int]]:
    """Depth-first exact search with state memoization and equivalent-child pruning."""
    env = CountdownEnv(list(numbers), target, require_use_all)
    expanded = 0
    duplicates = 0
    seen = set()

    def visit(state: Tuple[int, ...], path: Tuple[ActionKey, ...]):
        nonlocal expanded, duplicates
        if not require_use_all and target in state:
            return path
        if require_use_all and len(state) == 1 and state[0] == target:
            return path
        if len(state) < 2 or state in seen:
            if state in seen:
                duplicates += 1
            return None
        seen.add(state)
        expanded += 1
        child_states = set()
        for left, op, right, result, next_state in env.get_valid_actions(state):
            if next_state in child_states or next_state in seen:
                duplicates += 1
                continue
            child_states.add(next_state)
            answer = visit(next_state, path + ((left, op, right, result),))
            if answer is not None:
                return answer
        return None

    solution = visit(env.reset(), ())
    valid = int(solution is not None and verify_trajectory(numbers, target, solution, require_use_all).valid)
    exact = int(solution is not None and verify_trajectory(numbers, target, solution, require_use_all).success)
    return solution, {
        "states_expanded": expanded,
        "unique_states": len(seen),
        "duplicate_states": duplicates,
        "valid_solutions": valid,
        "exact_solutions": exact,
    }

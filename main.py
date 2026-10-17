import argparse
import random
import re
import sys
import time
from pathlib import Path

# Đảm bảo in ký tự tiếng Việt trên Windows console không bị lỗi mã hóa cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from matplotlib.patches import FancyBboxPatch
# pyrefly: ignore [missing-import]
from matplotlib.widgets import Button

from tree_visualizer import SearchTreeRecorder


INPUT_FILE = Path(__file__).with_name("input.txt")
OUTPUT_FILE = Path(__file__).with_name("ketqua.txt")
GRPO_CHECKPOINT_FILE = Path(__file__).with_name("grpo_checkpoint")
TREE_DOT_FILE = Path(__file__).with_name("tree.dot")
TREE_PNG_FILE = Path(__file__).with_name("tree.png")
OPERATOR_SYMBOLS = {"+": "+", "-": "−", "*": "×", "/": "÷"}


def read_problem(path):
    """Đọc các số đầu vào và giá trị Target từ file input.txt."""
    content = path.read_text(encoding="utf-8")
    target_match = re.search(r"\bTarget\s*:\s*(\d+)\b", content, re.IGNORECASE)
    if target_match is None:
        raise ValueError("Thiếu dòng Target: <số đích> trong input.txt.")

    numbers = [int(value) for value in re.findall(r"\d+", content[:target_match.start()])]
    target = int(target_match.group(1))
    if not 2 <= len(numbers) <= 8:
        raise ValueError("Cần nhập từ 2 đến 8 số trước dòng Target.")
    if any(number <= 0 for number in numbers) or target <= 0:
        raise ValueError("Các số và target phải là số nguyên dương.")
    return numbers, target


def solve_countdown(numbers, target, recorder=None, require_use_all=True, checkpoint=None):
    """Giải một puzzle bằng policy GRPO đã lưu."""
    from grpo_agent import solve_countdown_grpo
    return solve_countdown_grpo(
        numbers=numbers,
        recorder=recorder,
        target=target,
        require_use_all=require_use_all,
        checkpoint=str(checkpoint) if checkpoint else None,
    )


def build_steps(numbers, operations):
    """Dựng các bước tính toán và biểu thức toán học cuối cùng từ chuỗi hành động RL."""
    values = [(number, str(number)) for number in numbers]
    steps = []
    for left, operator, right, result in operations:
        before = tuple(sorted(value for value, _ in values))
        left_index = next(i for i, (value, _) in enumerate(values) if value == left)
        right_index = next(
            i for i, (value, _) in enumerate(values)
            if value == right and i != left_index
        )
        left_expression = values[left_index][1]
        right_expression = values[right_index][1]
        removed = {left_index, right_index}
        expression = f"({left_expression} {operator} {right_expression})"
        values = [item for i, item in enumerate(values) if i not in removed]
        values.append((result, expression))
        steps.append({
            "left": left,
            "operator": operator,
            "right": right,
            "result": result,
            "before": before,
            "after": tuple(sorted(value for value, _ in values)),
            "expression": expression,
        })
    return steps, values[0][1] if values else None


def save_results(numbers, target, steps, expression, elapsed, stats=None):
    """Lưu kết quả chạy và thông tin thuật toán RL vào file ketqua.txt."""
    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        output.write("=== COUNTDOWN — GRPO ===\n")
        output.write(f"Các số: {' '.join(map(str, numbers))}\n")
        output.write(f"Target: {target}\n")
        if stats:
            output.write(f"Thuật toán: {stats.get('algorithm', 'GRPO')}\n")
            if "trajectories_generated" in stats:
                output.write(f"Số trajectory GRPO đã sinh: {stats['trajectories_generated']}\n")
            if "policy_updates_at_load" in stats:
                output.write(f"Số lần cập nhật policy: {stats['policy_updates_at_load']}\n")
            if "solution_source" in stats:
                output.write(f"Nguồn nghiệm: {stats['solution_source'] or 'không có'}\n")
            if "states_expanded" in stats:
                output.write(f"State tìm kiếm mở rộng: {stats['states_expanded']}\n")
                output.write(f"State trùng đã bỏ qua: {stats['duplicate_states']}\n")
        if expression is None:
            output.write("Không tìm thấy lời giải chính xác.\n")
        else:
            output.write(f"Lời giải: {expression} = {target}\n")
            for index, step in enumerate(steps, start=1):
                symbol = OPERATOR_SYMBOLS[step["operator"]]
                output.write(
                    f"Bước {index}: {step['left']} {symbol} "
                    f"{step['right']} = {step['result']}\n"
                )
        output.write(f"Thời gian tìm kiếm: {elapsed * 1000:.3f} ms\n")


def save_experiment_report(lines, title="=== COUNTDOWN — GRPO THÍ NGHIỆM RL ==="):
    """Lưu báo cáo thí nghiệm RL (train/eval) vào ketqua.txt."""
    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        output.write(title + "\n")
        for line in lines:
            output.write(line + "\n")


def build_grpo_config(args):
    """ Ánh xạ tham số argparse -> GRPOConfig (module grpo_agent)."""
    from grpo_agent import GRPOConfig
    return GRPOConfig(
        learning_rate=args.lr,
        batch_size=args.grpo_batch_size,
        group_size=args.group_size,
        num_epochs=args.epochs,
        inner_epochs=args.inner_epochs,
        max_steps_per_episode=args.max_steps,
        clip_epsilon=args.clip_epsilon,
        kl_coefficient=args.kl_coeff,
        seed=args.seed,
        checkpoint_interval=args.ckpt_interval,
        evaluation_interval=args.eval_interval,
        device=args.device,
        max_gradient_norm=args.max_grad_norm,
        require_use_all=not args.allow_partial,
    )


def run_grpo_train(args):
    """Chế độ train (GRPO): sinh puzzle có nghiệm, chia train/val/test không giao
    nhau, huấn luyện GRPO (group + verifier reward + clipped surrogate) và lưu
    checkpoint RIÊNG vào thư mục grpo_checkpoint/ (không ghi đè q_table.json)."""
    from grpo_agent import GRPOAgent, make_data_splits

    ckpt = Path(args.grpo_file)
    config = build_grpo_config(args)

    print("=" * 60)
    print("   GRPO — HUẤN LUYỆN GROUP RELATIVE POLICY OPTIMIZATION")
    print("=" * 60)
    print(f"Đang sinh puzzle có nghiệm (train={args.train_instances}, "
          f"val={max(5, args.train_instances // 4)}, test={args.eval_instances})...")
    splits = make_data_splits(
        seed=args.seed,
        train_count=args.train_instances,
        val_count=max(5, args.train_instances // 4),
        test_count=args.eval_instances,
    )
    print(f"Splits: {len(splits['train'])} train / {len(splits['val'])} val / "
          f"{len(splits['test'])} test (không giao nhau).")
    print(f"Config: lr={config.learning_rate} group_size={config.group_size} "
          f"batch={config.batch_size} epochs={config.num_epochs} clip={config.clip_epsilon} "
          f"kl={config.kl_coefficient} device={config.device} seed={config.seed}")

    # Nạp checkpoint cũ (nếu có) để huấn luyện tăng dần — giống hành vi train Q-table.
    if (ckpt / "policy.pt").exists():
        agent = GRPOAgent.load_checkpoint(ckpt)
        print(f"Đã nạp checkpoint GRPO cũ: {ckpt} ({agent.policy_updates} cập nhật policy).")
        agent.config = config
        agent.optimizer = __import__("torch").optim.Adam(agent.policy.parameters(), lr=config.learning_rate)
    else:
        agent = GRPOAgent(config)
    agent.data_splits = splits

    start_time = time.perf_counter()
    result = agent.train_grpo(splits["train"], val_instances=splits["val"],
                              checkpoint_path=str(ckpt))
    elapsed = time.perf_counter() - start_time

    last = result["epochs"][-1] if result["epochs"] else {}
    lines = [
        "Chế độ: train (GRPO — Group Relative Policy Optimization)",
        f"Số puzzle train/val/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}",
        f"Group size G: {config.group_size} | Batch puzzle: {config.batch_size} | Epochs: {config.num_epochs}",
        f"Clip epsilon: {config.clip_epsilon} | KL beta: {config.kl_coefficient} | LR: {config.learning_rate}",
        f"Tổng số lần cập nhật policy (optimizer.step): {result['policy_updates']}",
        f"Epoch cuối: mean_reward={last.get('mean_reward', 0):+.3f} "
        f"train_solve={last.get('train_group_solve_rate', 0):.0%} "
        f"loss={last.get('loss', 0):+.4f} entropy={last.get('entropy', 0):.2f} "
        f"dup={last.get('duplicate_rate', 0):.0%} "
        f"val_exact={last.get('val_exact_solve_rate', float('nan')):.0%}",
        f"Thời gian huấn luyện: {elapsed:.1f} s",
        f"Đã lưu checkpoint GRPO vào: {ckpt}/ (policy.pt, reference.pt, optimizer.pt, meta.json)",
        "",
        f"Đánh giá held-out bằng: python main.py --mode eval --grpo-file {ckpt}",
    ]
    print("\n--- KẾT QUẢ HUẤN LUYỆN GRPO ---")
    for line in lines:
        print(line)
    save_experiment_report(lines, title="=== COUNTDOWN — GRPO THÍ NGHIỆM RL ===")
    print(f"\nĐã lưu báo cáo vào: {OUTPUT_FILE}")


def run_grpo_eval(args):
    """Chế độ eval (GRPO): nạp checkpoint, đánh giá policy THUẦN (greedy argmax,
    không heuristic) trên tập test held-out đã lưu trong checkpoint (hoặc sinh mới
    loại trừ tập train), kèm phương án best-of-N sampling để so sánh."""
    from grpo_agent import GRPOAgent, generate_solvable_instances

    ckpt = Path(args.grpo_file)
    if not (ckpt / "policy.pt").exists():
        print(f"Không tìm thấy checkpoint GRPO tại {ckpt}.")
        print("Hãy chạy huấn luyện trước: python main.py --mode train")
        sys.exit(1)
    agent = GRPOAgent.load_checkpoint(ckpt)

    test_instances = agent.data_splits.get("test") or []
    if not test_instances:
        rng = random.Random(args.eval_seed)
        test_instances = generate_solvable_instances(
            args.eval_instances, rng=rng, exclude=agent.train_instances
        )

    print("=" * 60)
    print("   GRPO — ĐÁNH GIÁ TỔNG QUÁT HÓA (HELD-OUT TEST, POLICY THUẦN)")
    print("=" * 60)
    print(f"Checkpoint: {ckpt} ({agent.policy_updates} cập nhật policy, "
          f"{len(agent.train_instances)} puzzle train).")
    print(f"Đang đánh giá greedy thuần trên {len(test_instances)} puzzle held-out...")

    start_time = time.perf_counter()
    greedy = agent.evaluate_grpo(test_instances, greedy=True, best_of_n=1)
    sampled = agent.evaluate_grpo(test_instances, greedy=False, best_of_n=args.best_of_n)
    elapsed = time.perf_counter() - start_time

    lines = [
        "Chế độ: eval (GRPO — policy thuần từ checkpoint, nghiệm đã qua verifier độc lập)",
        f"Nguồn checkpoint: {ckpt} ({agent.policy_updates} lần cập nhật tham số)",
        f"Số puzzle test held-out: {greedy['instances']} (không trùng tập train)",
        "",
        "[1] Policy greedy THUẦN (argmax π_θ, không sampling, không heuristic):",
        f"    Exact solve rate:     {greedy['exact_solve_rate']:.1%}",
        f"    Valid solution rate:  {greedy['valid_solution_rate']:.1%}",
        f"    Mean abs error:       {greedy['mean_abs_error']:.1f}",
        f"    Mean traj length:     {greedy['mean_traj_length']:.2f}",
        "",
        f"[2] Best-of-{args.best_of_n} sampling (lấy mẫu N nghiệm, chọn nghiệm tốt nhất đã verifier):",
        f"    Exact solve rate:     {sampled['exact_solve_rate']:.1%}",
        f"    Valid solution rate:  {sampled['valid_solution_rate']:.1%}",
        f"    Mean abs error:       {sampled['mean_abs_error']:.1f}",
        f"    Mean traj length:     {sampled['mean_traj_length']:.2f}",
        "",
        f"Thời gian đánh giá: {elapsed:.1f} s",
    ]

    # Đánh giá thêm trên puzzle hiện tại trong input.txt (nếu đọc được).
    try:
        numbers, target = read_problem(INPUT_FILE)
        ops, puzzle_stats = solve_countdown(
            numbers=numbers,
            target=target,
            require_use_all=not args.allow_partial,
            checkpoint=ckpt,
        )
        status = "ĐẠT (đã verifier xác nhận)" if ops is not None else "KHÔNG (không có nghiệm được xác nhận)"
        lines += [
            "",
            f"Puzzle trong input.txt: {' '.join(map(str, numbers))} | Target {target} -> {status} "
            f"[nguồn: {puzzle_stats.get('solution_source') or 'không có'}]",
            "(Lưu ý: để giải input.txt, dùng: python main.py --mode solve)",
        ]
    except (OSError, ValueError):
        pass

    print()
    for line in lines:
        print(line)
    save_experiment_report(lines, title="=== COUNTDOWN — GRPO THÍ NGHIỆM RL ===")
    print(f"\nĐã lưu báo cáo vào: {OUTPUT_FILE}")


def show_simulation(numbers, target, steps, expression):
    """Giao diện đồ họa hiển thị từng bước giải mô phỏng tương tác bằng matplotlib."""
    figure, axis = plt.subplots(figsize=(11, 7))
    figure.patch.set_facecolor("#f3f4ef")
    axis.set_facecolor("#f3f4ef")
    plt.subplots_adjust(bottom=0.15)
    step_index = {"value": 0}
    total_steps = len(steps)

    def draw_frame():
        axis.clear()
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.text(0.06, 0.94, "COUNTDOWN (GRPO SOLVER)", fontsize=22, weight="bold", color="#17324d")
        axis.text(0.94, 0.945, f"TARGET  {target}", ha="right", fontsize=15,
                  weight="bold", color="#c65d3a")
        axis.text(0.06, 0.875, f"SỐ BAN ĐẦU   {'  ·  '.join(map(str, numbers))}",
                  fontsize=11, color="#52616b")

        current_index = step_index["value"]
        if expression is None:
            axis.text(0.5, 0.55, "Không tìm thấy lời giải chính xác",
                      ha="center", va="center", fontsize=20, color="#c65d3a")
            return

        axis.text(0.06, 0.79, f"CÁC SỐ ĐANG CÓ   ·   BƯỚC {current_index}/{total_steps}",
                  fontsize=11, weight="bold", color="#52616b")
        operation = steps[current_index - 1] if current_index else None
        current_values = operation["before"] if operation else tuple(sorted(numbers))
        selected = []
        if operation:
            selected = [operation["left"], operation["right"]]

        card_width = min(0.13, 0.78 / max(len(current_values), 1))
        gap = 0.018
        total_width = len(current_values) * card_width + (len(current_values) - 1) * gap
        first_x = (1 - total_width) / 2
        for index, value in enumerate(current_values):
            x = first_x + index * (card_width + gap)
            color = "#e9a23b" if value in selected else "#ffffff"
            if value in selected:
                selected.remove(value)
            card = FancyBboxPatch(
                (x, 0.65), card_width, 0.095,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=color, edgecolor="#d6dedc", linewidth=1.2,
            )
            axis.add_patch(card)
            axis.text(x + card_width / 2, 0.697, str(value), ha="center", va="center",
                      fontsize=16, weight="bold", color="#17324d")

        axis.text(0.5, 0.52, "PHÉP TÍNH", ha="center", fontsize=10,
                  weight="bold", color="#718096")
        if operation:
            equation = (
                f"{operation['left']} {OPERATOR_SYMBOLS[operation['operator']]} "
                f"{operation['right']}  =  {operation['result']}"
            )
            axis.text(0.5, 0.41, equation, ha="center", va="center", fontsize=25,
                      weight="bold", color="#176b63")
            axis.text(0.06, 0.29, "CÒN LẠI SAU BƯỚC NÀY", fontsize=10,
                  weight="bold", color="#52616b")
            after_values = operation["after"]
            result_width = min(0.13, 0.78 / max(len(after_values), 1))
            result_gap = 0.018
            result_total = len(after_values) * result_width + (len(after_values) - 1) * result_gap
            result_x = (1 - result_total) / 2
            for index, value in enumerate(after_values):
                x = result_x + index * (result_width + result_gap)
                is_result = value == operation["result"]
                card = FancyBboxPatch(
                    (x, 0.17), result_width, 0.09,
                    boxstyle="round,pad=0.012,rounding_size=0.018",
                    facecolor="#176b63" if is_result else "#ffffff",
                    edgecolor="#176b63" if is_result else "#d6dedc", linewidth=1.2,
                )
                axis.add_patch(card)
                axis.text(x + result_width / 2, 0.215, str(value), ha="center", va="center",
                          fontsize=15, weight="bold",
                          color="#ffffff" if is_result else "#17324d")
            axis.text(0.5, 0.105, operation["expression"], ha="center", va="center",
                      fontsize=11, color="#52616b", wrap=True)
            if current_index == total_steps:
                axis.text(0.5, 0.035, "ĐÃ ĐẠT TARGET", ha="center", fontsize=11,
                          weight="bold", color="#176b63")
        else:
            axis.text(0.5, 0.41, "Sẵn sàng bắt đầu", ha="center", va="center",
                      fontsize=22, weight="bold", color="#176b63")
            axis.text(0.5, 0.28, "Nhấn Tiếp để xem phép tính đầu tiên",
                      ha="center", fontsize=12, color="#52616b")
        figure.canvas.draw_idle()

    def move_step(amount):
        step_index["value"] = max(0, min(total_steps, step_index["value"] + amount))
        draw_frame()

    previous_axis = figure.add_axes((0.34, 0.035, 0.14, 0.065))
    next_axis = figure.add_axes((0.52, 0.035, 0.14, 0.065))
    previous_button = Button(previous_axis, "← Trước", color="#ffffff", hovercolor="#e3ece8")
    next_button = Button(next_axis, "Tiếp →", color="#ffffff", hovercolor="#e3ece8")
    previous_button.on_clicked(lambda _event: move_step(-1))
    next_button.on_clicked(lambda _event: move_step(1))
    draw_frame()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Countdown Solver chỉ sử dụng GRPO.")
    parser.add_argument(
        "--mode",
        choices=["solve", "train", "eval"],
        default="solve",
        help="solve: giải input.txt; train: huấn luyện policy GRPO; eval: đánh giá GRPO trên test held-out.",
    )
    parser.add_argument(
        "--grpo-file",
        default=str(GRPO_CHECKPOINT_FILE),
        help="Thư mục checkpoint GRPO (policy.pt/reference.pt/optimizer.pt/meta.json) "
             "Mặc định: grpo_checkpoint/.",
    )
    parser.add_argument(
        "--train-instances",
        type=int,
        default=100,
        help="Số instance huấn luyện khi --mode train (mặc định 100).",
    )
    parser.add_argument(
        "--eval-instances",
        type=int,
        default=50,
        help="Số puzzle held-out khi --mode eval (mặc định 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed sinh tập instance huấn luyện (--mode train) và torch/random cho GRPO.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Seed sinh tập puzzle held-out (--mode eval).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Chạy không mở cửa sổ giao diện Matplotlib (chỉ áp dụng chế độ solve).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Cho phép dừng ngay khi đạt target mà không cần dùng hết toàn bộ số đầu vào.",
    )
    # -------------------- Tham số GRPO --------------------
    grpo_group = parser.add_argument_group(
        "GRPO", "Tham số huấn luyện và đánh giá GRPO."
    )
    grpo_group.add_argument("--lr", type=float, default=3e-3, help="Learning rate GRPO (mặc định 3e-3).")
    grpo_group.add_argument("--grpo-batch-size", type=int, default=4,
                            help="Số puzzle mỗi lần sinh group + cập nhật GRPO (mặc định 4).")
    grpo_group.add_argument("--group-size", type=int, default=8,
                            help="G — số trajectory lấy mẫu cho MỖI puzzle trong một group (mặc định 8).")
    grpo_group.add_argument("--epochs", type=int, default=10,
                            help="Số epoch huấn luyện GRPO (--mode train, mặc định 10).")
    grpo_group.add_argument("--inner-epochs", type=int, default=2,
                            help="Số lần tái dùng batch đã sinh cho gradient (PPO-style, mặc định 2).")
    grpo_group.add_argument("--max-steps", type=int, default=16,
                            help="Trần số bước hành động mỗi trajectory GRPO (mặc định 16).")
    grpo_group.add_argument("--clip-epsilon", type=float, default=0.2,
                            help="ε clipped surrogate objective GRPO (mặc định 0.2).")
    grpo_group.add_argument("--kl-coeff", type=float, default=0.0,
                            help="β hệ số KL(π_θ||π_ref); 0 = tắt reference policy (mặc định 0.0).")
    grpo_group.add_argument("--max-grad-norm", type=float, default=1.0,
                            help="Clip tổng norm gradient (mặc định 1.0).")
    grpo_group.add_argument("--device", default="cpu", help="Thiết bị torch cho GRPO (mặc định 'cpu').")
    grpo_group.add_argument("--ckpt-interval", type=int, default=2,
                            help="Lưu checkpoint GRPO mỗi N epoch (mặc định 2).")
    grpo_group.add_argument("--eval-interval", type=int, default=2,
                            help="Đánh giá tập validation mỗi N epoch (mặc định 2).")
    grpo_group.add_argument("--best-of-n", type=int, default=16,
                            help="Số trajectory lấy mẫu best-of-N khi --mode eval (mặc định 16).")
    args = parser.parse_args()

    if args.mode == "train":
        run_grpo_train(args)
        return
    if args.mode == "eval":
        run_grpo_eval(args)
        return

    try:
        numbers, target = read_problem(INPUT_FILE)
    except (OSError, ValueError) as error:
        print(f"Lỗi input: {error}")
        sys.exit(1)

    tree_recorder = SearchTreeRecorder(target, algorithm="GRPO sampled rollout")

    print("=" * 60)
    print("   COUNTDOWN SOLVER — GRPO + EXACT SEARCH FALLBACK")
    print("=" * 60)
    print(f"Các số ban đầu: {numbers}")
    print(f"Mục tiêu Target: {target}")
    print(f"Bắt buộc dùng hết số: {not args.allow_partial}")
    print("Đang thử policy GRPO; nếu chưa giải được sẽ chạy exact-search baseline...")

    start_time = time.perf_counter()
    try:
        operations, stats = solve_countdown(
            numbers=numbers,
            target=target,
            recorder=tree_recorder,
            require_use_all=not args.allow_partial,
            checkpoint=args.grpo_file,
        )
    except FileNotFoundError as error:
        print(f"Lỗi: {error}")
        sys.exit(1)
    elapsed = time.perf_counter() - start_time
    steps, expression = build_steps(numbers, operations) if operations is not None else ([], None)
    # Kiểm chứng lại kết quả GRPO trước khi công bố.
    if operations is not None:
        from grpo_agent import verify_trajectory
        if not verify_trajectory(numbers, target, operations, not args.allow_partial).success:
            print("CẢNH BÁO: lời giải không vượt qua verifier độc lập — coi như không hợp lệ.")
            operations, steps, expression = None, [], None

    dot_path = tree_recorder.write_dot(TREE_DOT_FILE)
    tree_recorder.render_png(dot_path, TREE_PNG_FILE)

    print("\n--- KẾT QUẢ SOLVER ---")
    print(f"Thuật toán: {stats.get('algorithm', 'GRPO')}")
    print(f"Nguồn lời giải: {stats.get('solution_source') or 'không tìm thấy'}")
    print(f"GRPO tự giải được: {'có' if stats.get('grpo_verified') else 'không'}")
    if "trajectories_generated" in stats:
        print(f"Số trajectory GRPO đã sinh: {stats['trajectories_generated']}")
        print(f"Số lần cập nhật policy trong checkpoint: {stats.get('policy_updates_at_load', 0)}")
    if "states_expanded" in stats:
        print(f"State exact search đã mở rộng: {stats['states_expanded']}")
        print(f"Node trạng thái duy nhất: {stats['unique_states']}")
        print(f"State trùng đã bỏ qua: {stats['duplicate_states']}")
        print(f"Lời giải hợp lệ/chính xác từ search: {stats['valid_solutions']}/{stats['exact_solutions']}")
    print(f"Số node cây GRPO (depth 0-2): {len(tree_recorder.nodes)}")
    print(f"Số cạnh cây GRPO (depth 0-2): {len(tree_recorder.edges)}")
    print(f"Thời gian tìm kiếm: {elapsed * 1000:.3f} ms")

    print("\n--- LỜI GIẢI ---")
    if expression is None:
        print("Không tìm thấy lời giải chính xác.")
    else:
        print(f"Biểu thức: {expression} = {target}")
        for index, step in enumerate(steps, start=1):
            print(
                f"Bước {index}: {step['left']} {OPERATOR_SYMBOLS[step['operator']]} "
                f"{step['right']} = {step['result']}"
            )

    save_results(numbers, target, steps, expression, elapsed, stats=stats)
    print(f"\nĐã lưu kết quả vào: {OUTPUT_FILE}")
    print(f"Đã tạo cây trajectory GRPO: {dot_path}")

    if not args.no_gui:
        show_simulation(numbers, target, steps, expression)


if __name__ == "__main__":
    main()

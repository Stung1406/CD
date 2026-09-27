import sys
import time
from collections import defaultdict
import networkx as nx
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button


INPUT_FILE = Path(__file__).with_name("input.txt")
OUTPUT_FILE = Path(__file__).with_name("ketqua.txt")
OPERATOR_SYMBOLS = {"+": "+", "-": "−", "*": "×", "/": "÷"}


def read_problem(path):
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


def countdown(numbers, target):
    @lru_cache(maxsize=None)
    def search(state):
        if len(state) == 1:
            return () if state[0] == target else None

        for first_index in range(len(state)):
            for second_index in range(first_index + 1, len(state)):
                first, second = state[first_index], state[second_index]
                remaining = [
                    value for index, value in enumerate(state)
                    if index not in (first_index, second_index)
                ]
                candidates = [
                    (first, "+", second, first + second),
                    (first, "*", second, first * second),
                ]
                if first > second:
                    candidates.append((first, "-", second, first - second))
                elif second > first:
                    candidates.append((second, "-", first, second - first))
                if first % second == 0:
                    candidates.append((first, "/", second, first // second))
                if second % first == 0:
                    candidates.append((second, "/", first, second // first))

                tried_results = set()
                for left, operator, right, result in candidates:
                    if result <= 0 or result in tried_results:
                        continue
                    tried_results.add(result)
                    next_state = tuple(sorted((*remaining, result)))
                    later_steps = search(next_state)
                    if later_steps is not None:
                        return ((left, operator, right, result),) + later_steps
        return None

    return search(tuple(sorted(numbers)))


def build_steps(numbers, operations):
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


def save_results(numbers, target, steps, expression, elapsed):
    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        output.write("=== COUNTDOWN ===\n")
        output.write(f"Các số: {' '.join(map(str, numbers))}\n")
        output.write(f"Target: {target}\n")
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


def show_simulation(numbers, target, steps, expression):
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
        axis.text(0.06, 0.94, "COUNTDOWN", fontsize=23, weight="bold", color="#17324d")
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
    try:
        numbers, target = read_problem(INPUT_FILE)
    except (OSError, ValueError) as error:
        print(f"Lỗi input: {error}")
        sys.exit(1)

    start_time = time.perf_counter()
    operations = countdown(numbers, target)
    elapsed = time.perf_counter() - start_time
    steps, expression = build_steps(numbers, operations) if operations is not None else ([], None)

    print(f"Các số: {numbers} | Target: {target}")
    if expression is None:
        print("Không tìm thấy lời giải chính xác.")
    else:
        print(f"Lời giải: {expression} = {target}")
        for index, step in enumerate(steps, start=1):
            print(
                f"Bước {index}: {step['left']} {OPERATOR_SYMBOLS[step['operator']]} "
                f"{step['right']} = {step['result']}"
            )
    print(f"Thời gian tìm kiếm: {elapsed * 1000:.3f} ms")
    save_results(numbers, target, steps, expression, elapsed)
    print(f"Đã lưu kết quả vào: {OUTPUT_FILE}")
    show_simulation(numbers, target, steps, expression)


if __name__ == "__main__":
    main()
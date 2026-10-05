"""Ghi lại và xuất các tầng đầu của cây tìm kiếm RL (MCTS/Q-learning) Countdown."""

import shutil
import subprocess
from pathlib import Path


class SearchTreeRecorder:
    """Lưu các chuyển trạng thái solver thực sự thử ở độ sâu 0..2 kèm chỉ số RL."""

    MAX_DEPTH = 2

    def __init__(self, target, algorithm="RL (MCTS)"):
        self.target = target
        self.algorithm = algorithm
        self.nodes = []
        self.edges = []
        self._node_by_id = {}
        self._edge_by_target = {}
        self.root_id = None

    def start(self, state, visits=None, q_value=None):
        """Tạo node gốc từ trạng thái ban đầu đã sắp xếp."""
        self.root_id = self._add_node(state, depth=0, visits=visits, q_value=q_value)
        return self.root_id

    def add_transition(self, parent_id, state, depth, operation, visits=None, q_value=None):
        """Ghi cạnh và node con ngay trước khi solver duyệt tiếp."""
        if parent_id is None or depth > self.MAX_DEPTH:
            return None

        child_id = self._add_node(state, depth, visits=visits, q_value=q_value)
        left, operator, right, result = operation
        edge = {
            "source": parent_id,
            "target": child_id,
            "label": f"{left} {operator} {right} = {result}",
            "status": "SEARCHING",
        }
        self.edges.append(edge)
        self._edge_by_target[child_id] = edge
        return child_id

    def update_node_rl(self, node_id, visits, q_value):
        """Cập nhật số lần ghé thăm và giá trị Q-value cho node."""
        if node_id is not None and node_id in self._node_by_id:
            self._node_by_id[node_id]["visits"] = visits
            self._node_by_id[node_id]["q_value"] = q_value

    def finish_node(self, node_id, succeeded, visits=None, q_value=None):
        """Đánh dấu kết quả sau khi tìm kiếm xong node."""
        if node_id is None:
            return

        status = "SUCCESS" if succeeded else "FAILED"
        self._node_by_id[node_id]["status"] = status
        if visits is not None:
            self._node_by_id[node_id]["visits"] = visits
        if q_value is not None:
            self._node_by_id[node_id]["q_value"] = q_value

        edge = self._edge_by_target.get(node_id)
        if edge is not None:
            edge["status"] = status

    def finish(self, succeeded, visits=None, q_value=None):
        """Đánh dấu trạng thái cuối cùng của node gốc."""
        self.finish_node(self.root_id, succeeded, visits=visits, q_value=q_value)

    def write_dot(self, dot_path):
        """Ghi cây đã thu thập thành tệp Graphviz DOT kèm chỉ số RL."""
        dot_path = Path(dot_path)
        lines = [
            "digraph CountdownRL {",
            (
                '    graph [rankdir="TB", ordering="out", labelloc="t", '
                f'label="Countdown {self.algorithm} | Target: {self.target} | Depth: 0..{self.MAX_DEPTH}"];'
            ),
            '    node [shape="box", style="rounded,filled", fontname="Arial"];',
            '    edge [fontname="Arial"];',
        ]

        for node in self.nodes:
            state_label = "[" + ", ".join(map(str, node["state"])) + "]"
            rl_info = ""
            if node.get("visits") is not None:
                q_val = node.get("q_value", 0.0)
                rl_info = f"\nN={node['visits']} | Q={q_val:+.2f}"
            label = f"{state_label}\ndepth={node['depth']}{rl_info}\n{node['status']}"
            fill_color = {
                "SUCCESS": "#d9f2df",
                "FAILED": "#f8dddd",
                "SEARCHING": "#fff1c2",
            }[node["status"]]
            lines.append(
                f'    {node["id"]} [label="{self._escape(label)}", '
                f'fillcolor="{fill_color}"];'
            )

        for edge in self.edges:
            color = "#25824b" if edge["status"] == "SUCCESS" else "#bd4545"
            label = f"{edge['label']}\n{edge['status']}"
            lines.append(
                f'    {edge["source"]} -> {edge["target"]} '
                f'[label="{self._escape(label)}", color="{color}"];'
            )

        lines.append("}")
        dot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dot_path

    def render_png(self, dot_path, png_path):
        """Tự render PNG nếu tìm thấy Graphviz; DOT luôn được giữ lại."""
        dot_executable = shutil.which("dot")
        if dot_executable is None:
            print(
                "Graphviz chưa được cài hoặc chưa có trong PATH. "
                "Tệp tree.dot đã được tạo; cài Graphviz từ "
                "https://graphviz.org/download/ rồi chạy lại để tạo tree.png."
            )
            return False

        try:
            subprocess.run(
                [dot_executable, "-Tpng", str(dot_path), "-o", str(png_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            print(f"Không thể render tree.png; tree.dot vẫn sẵn sàng. Chi tiết: {detail}")
            return False

        print(f"Đã tạo ảnh cây tìm kiếm: {png_path}")
        return True

    def _add_node(self, state, depth, visits=None, q_value=None):
        node_id = f"n{len(self.nodes)}"
        node_data = {
            "id": node_id,
            "state": tuple(state),
            "depth": depth,
            "status": "SEARCHING",
        }
        if visits is not None:
            node_data["visits"] = visits
        if q_value is not None:
            node_data["q_value"] = q_value

        self.nodes.append(node_data)
        self._node_by_id[node_id] = self.nodes[-1]
        return node_id

    @staticmethod
    def _escape(value):
        return (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )


import math
from dataclasses import dataclass


@dataclass
class Node:
    name: str
    cap_gb: float
    fixed_gb: float
    sec_per_layer: float
    max_layers: int | None = None
    layer_gb: float | None = None
    min_layers: int = 0
    link_sec: float = 0.0
    spill_gb: float = 0.0
    spill_layer_gb: float = 0.0
    spill_sec_per_layer: float = 0.0

    def on_gpu(self, count: int, layer_gb: float) -> int:
        """How many of `count` layers the device's own memory holds; with spill_gb, the rest go to its RAM (its CPU runs them)."""
        if not self.spill_gb:
            return count
        per = self.layer_gb if self.layer_gb is not None else layer_gb
        room = int(max(0.0, self.cap_gb - self.fixed_gb) / per) if per > 0 else count
        return min(count, room)

    def fits(self, count: int, layer_gb: float) -> bool:
        if self.max_layers is not None and count > self.max_layers or count < self.min_layers:
            return False
        per = self.layer_gb if self.layer_gb is not None else layer_gb
        fast = self.on_gpu(count, layer_gb)
        if self.fixed_gb + fast * per > self.cap_gb:
            return False
        return (count - fast) * self.spill_layer_gb <= self.spill_gb

    def seconds(self, count: int, layer_gb: float) -> float:
        """Time per step: the GPU part and the RAM part work as a pipeline, so the slower of the two sets the pace."""
        fast = self.on_gpu(count, layer_gb)
        return max(self.sec_per_layer * fast, self.spill_sec_per_layer * (count - fast)) + self.link_sec


@dataclass
class Stage:
    name: str
    first: int
    last: int
    gb: float
    sec: float
    on_gpu: int | None = None


def layer_bounds(spec, n_layers: int) -> tuple[int, int | None]:
    """A device's layer setting as (fewest, most): "auto" or None, a count ("8"), a share of the model ("30%") or a range ("4-10")."""
    if spec in (None, "", "auto"):
        return 0, None
    text = str(spec).strip()
    if text.endswith("%"):
        exact = n_layers * float(text[:-1]) / 100
        return max(1, math.floor(exact)), max(1, math.ceil(exact))
    if "-" in text:
        low, high = (int(x) for x in text.split("-", 1))
        return low, high
    return int(text), int(text)


def plan(nodes: list[Node], n_layers: int, layer_gb: float) -> list[Stage] | None:
    """Split the layers so the slowest stage (its compute plus its sending) is as fast as possible."""
    n = len(nodes)
    if n == 0 or n > n_layers:
        return None

    def fits(node: Node, count: int) -> bool:
        return node.fits(count, layer_gb)

    inf = float("inf")
    best = [[inf] * (n_layers + 1) for _ in range(n + 1)]
    choice = [[0] * (n_layers + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(1, n + 1):
        node = nodes[i - 1]
        for used in range(i, n_layers - (n - i) + 1):
            for count in range(1, used - (i - 1) + 1):
                prev = best[i - 1][used - count]
                if prev == inf or not fits(node, count):
                    continue
                cost = max(prev, node.seconds(count, layer_gb))
                if cost < best[i][used]:
                    best[i][used] = cost
                    choice[i][used] = count
    if best[n][n_layers] == inf:
        return None
    counts, used = [], n_layers
    for i in range(n, 0, -1):
        counts.append(choice[i][used])
        used -= choice[i][used]
    counts.reverse()
    stages, first = [], 0
    for node, count in zip(nodes, counts):
        fast = node.on_gpu(count, layer_gb)
        gb = node.fixed_gb + fast * (layer_gb if node.layer_gb is None else node.layer_gb)
        stages.append(Stage(node.name, first, first + count, round(gb, 2), round(node.seconds(count, layer_gb), 3),
                            fast if node.spill_gb else None))
        first += count
    return stages

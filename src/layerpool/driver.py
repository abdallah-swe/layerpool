"""The machine with the training code runs the model's own forward. Its blocks are stand-ins: each one notes the arguments
the model gave it, and the last one sends the stack input with all those arguments through the devices and returns what comes
back. Autograd sees the whole trip as one operation, so loss.backward() sends the gradient back the same way."""
import math
import queue
import threading

import numpy as np

from .blocks import pack_args, unpack_args
from .codec import decode, encode, pack, split_prefix, unpack, with_prefix


class Box:
    """One trip through the devices: what was sent, and what is needed to bring the gradients back."""

    def __init__(self, ends, hidden, per_block, sides, hidden_kw):
        self.ends, self.hidden, self.per_block, self.sides, self.hidden_kw = ends, hidden, per_block, sides, hidden_kw
        self.adapter = ends.adapter
        self.train = False
        self.state = None
        self.dropped = False


def _remote_function():
    import torch

    class Remote(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, box, anchor, *sides):
            ctx.box = box
            return box.ends.chain.forward(box, hidden.detach()).cpu()

        @staticmethod
        def backward(ctx, grad):
            box = ctx.box
            if box.dropped:
                raise RuntimeError("layerpool: this output was dropped because another forward, an optimizer step or a save came before its "
                                   "backward(). Call backward() on each training output before the next forward.")
            g_hidden, g_sides, norm = box.ends.chain.backward(box, grad)
            anchor = box.ends.anchor
            before = float(anchor.grad.item()) if anchor.grad is not None else 0.0
            box.ends.remote_norm = norm
            g_sides = [None if g is None else g.cpu() for g in g_sides]
            return (None if g_hidden is None else g_hidden.cpu(), None, torch.tensor([norm - before]), *g_sides)
    return Remote


_REMOTE = None


def _stand_in_class():
    import torch

    class StandIn(torch.nn.Module):
        def __init__(self, index: int):
            super().__init__()
            self.index = index

        def forward(self, *args, **kwargs):
            ends = self.ends
            hidden_kw = None
            if args:
                hidden, rest = args[0], tuple(args[1:])
            else:
                hidden_kw = "hidden_states" if "hidden_states" in kwargs else next(k for k, v in kwargs.items() if torch.is_tensor(v))
                kwargs = dict(kwargs)
                hidden, rest = kwargs.pop(hidden_kw), ()
            call = ends.call
            if self.index == 0:
                ends.markers.clear()
                ends.keep.clear()
                call.clear()
                call.update(input=hidden, blocks={}, hidden_kw=hidden_kw)
            call["blocks"][self.index] = (rest, kwargs)
            if ends.tuple_out is None:
                ends.tuple_out = ends.probe(hidden, rest, kwargs, hidden_kw)
            if self.index < ends.n - 1:
                return ends.shaped(hidden, self.index)
            return ends.shaped(ends.through(call), self.index)
    return StandIn


class Ends:
    """The model on this machine, with its stack replaced by stand-ins (see blocks.load_ends)."""

    def __init__(self, model, lay: dict, chain):
        import torch
        self.model, self.lay, self.chain = model, lay, chain
        self.n = lay["n"]
        self.local = threading.local()
        self.tuple_out = None
        self.markers, self.keep = {}, []
        self.adapter = True
        self.remote_norm = 0.0
        self.discard = None
        path = lay["stack"]
        loaded = list(model.get_submodule(path))
        self._probe_block = loaded[0]
        picked = model.__dict__.get("_layerpool_picked", list(range(len(loaded))))
        kinds = model.__dict__.get("_layerpool_kinds", {})
        StandIn = _stand_in_class()
        stand_ins = torch.nn.ModuleList([StandIn(i) for i in range(self.n)])
        for i, s in enumerate(stand_ins):
            object.__setattr__(s, "ends", self)
            like = loaded[picked.index(kinds.get(i, picked[0]))] if kinds else loaded[0]
            for key, value in like.__dict__.items():
                if not key.startswith("_") and key not in ("training", "index") and isinstance(value, (str, int, float, bool, type(None))):
                    s.__dict__[key] = value
            if "layer_idx" in s.__dict__:
                s.__dict__["layer_idx"] = i
        parent_path, _, attr = path.rpartition(".")
        setattr(model.get_submodule(parent_path) if parent_path else model, attr, stand_ins)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        model.register_parameter("layerpool_anchor", self.anchor)

    @property
    def call(self) -> dict:
        if not hasattr(self.local, "call"):
            self.local.call = {}
        return self.local.call

    def shaped(self, hidden, index: int):
        """The hidden state in the form the model's blocks return it: alone, or first in a tuple or list as long as theirs. The
        other places hold None, or a marker where a real block returns a tensor: when the model hands that tensor to a later
        block (T5's position bias), the block's arguments name it ("output k of block i") and the devices put the real one there."""
        kind, length, tensors = self.tuple_out
        if kind is None:
            return hidden
        import torch
        extras = []
        for k in range(1, length):
            if k in tensors:
                marker = torch.empty(0)
                self.markers[id(marker)] = (index, k)
                self.keep.append(marker)
                extras.append(marker)
            else:
                extras.append(None)
        return kind([hidden] + extras)

    def probe(self, hidden, rest, kwargs, hidden_kw) -> tuple:
        """How blocks of this model return their output, from one real block run on this input (its weights do not matter)."""
        import torch
        block = self._probe_block
        if block is None:
            return (None, 1, set())
        try:
            with torch.no_grad():
                param = next(block.parameters())
                h = hidden.detach().to(param.device, param.dtype)
                kwargs = {k: (v.to(param.device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
                rest = tuple(v.to(param.device) if torch.is_tensor(v) else v for v in rest)
                out = block(**{hidden_kw: h}, **kwargs) if hidden_kw else block(h, *rest, **kwargs)
            if isinstance(out, (tuple, list)):
                return (type(out), len(out), {k for k, v in enumerate(out) if k > 0 and torch.is_tensor(v)})
            return (None, 1, set())
        except Exception:
            return (None, 1, set())
        finally:
            self._probe_block = None

    def through(self, call):
        global _REMOTE
        import torch
        if sorted(call["blocks"]) != list(range(self.n)):
            raise RuntimeError(f"the model ran {len(call['blocks'])} of its {self.n} blocks in this forward pass; layerpool needs "
                               "every block run once, in order (LayerDrop or skipped blocks are not supported)")
        hidden = call["input"]
        if self.discard is not None and self.remote_norm and torch.is_grad_enabled() and self.model.training and \
                (self.anchor.grad is None or not float(self.anchor.grad.abs().sum())):
            self.discard()
        sides, seen = [], set()

        def collect(v):
            if torch.is_tensor(v):
                if v.requires_grad and id(v) not in seen:
                    seen.add(id(v))
                    sides.append(v)
            elif isinstance(v, (tuple, list)):
                for x in v:
                    collect(x)
            elif isinstance(v, dict):
                for x in v.values():
                    collect(x)
        if torch.is_grad_enabled():
            for a, kw in call["blocks"].values():
                collect(a)
                collect(kw)
        box = Box(self, hidden, dict(call["blocks"]), sides, call["hidden_kw"])
        box.carried = dict(self.markers)
        box.train = torch.is_grad_enabled()
        if box.train:
            if _REMOTE is None:
                _REMOTE = _remote_function()
            out = _REMOTE.apply(hidden.cpu(), box, self.anchor.cpu(), *(s.cpu() for s in sides))
            return out.to(hidden.device)
        with torch.no_grad():
            return self.chain.forward(box, hidden.detach())


def _split_points(rows: int, micro: int, micro_rows: int | None) -> list[tuple[int, int]]:
    per = micro_rows or max(1, math.ceil(rows / max(1, micro)))
    return [(a, min(rows, a + per)) for a in range(0, rows, per)]


def split_args(per_block: dict, rows: int, cuts: list[tuple[int, int]]):
    """The block arguments of each micro-batch: tensors with the batch on their first dimension (or on the second, after a small
    first one, like M-RoPE positions) are cut by rows; everything else is shared. Returns the per-micro trees and, for every cut
    tensor, where it came from."""
    import torch
    if len(cuts) == 1:
        return [per_block], [{}]
    out, origins = [], []
    for a, b in cuts:
        made, origin = {}, {}

        def cut(v, a=a, b=b, made=made, origin=origin):
            if torch.is_tensor(v):
                if id(v) in made:
                    return made[id(v)]
                if v.dim() > 0 and v.shape[0] == rows:
                    piece, dim = v[a:b], 0
                elif v.dim() > 1 and v.shape[1] == rows and 1 < v.shape[0] <= 4:
                    piece, dim = v[:, a:b], 1
                else:
                    piece, dim = v, None
                made[id(v)] = piece
                origin[id(piece)] = (v, dim, a, b)
                return piece
            if isinstance(v, tuple):
                return tuple(cut(x) for x in v)
            if isinstance(v, list):
                return [cut(x) for x in v]
            if isinstance(v, dict):
                return {k: cut(x) for k, x in v.items()}
            return v
        out.append({i: (tuple(cut(x) for x in args), {k: cut(x) for k, x in kw.items()}) for i, (args, kw) in per_block.items()})
        origins.append(origin)
    return out, origins


class PoolChain:
    """The devices of a pool: every batch is cut into micro-batches that are all in flight at once."""

    def __init__(self, pool, job: dict):
        self.pool, self.job = pool, job
        self.io_codec = "fp32" if job.get("codec") == "fp32" else "fp16"
        self.pending: Box | None = None

    def drop_pending(self, redo: bool = False):
        """Finish the operation that waits for a backward. With redo, its backward can still come later (another forward came
        first, as with a contrastive loss over two batches): it then runs again with the gradient it gets. Otherwise (an optimizer
        step or a save came first) a later backward raises."""
        box, self.pending = self.pending, None
        if box is not None and box.state is not None and not box.state["done"].is_set():
            box.state["cots"].put(None)
            box.state["done"].wait()
            if redo:
                box.state["redo"] = True
            else:
                box.dropped = True

    def pieces(self, rows: int) -> tuple[int, int]:
        """(rows of the biggest piece, number of pieces) a batch of `rows` rows is cut into."""
        cuts = _split_points(rows, self.job.get("micro", 8), self.job.get("micro_batch_size"))
        return max(b - a for a, b in cuts), len(cuts)

    def expect(self, rows: int, seq: int, pieces: int | None = None):
        """Pieces of `rows` rows of up to `seq` tokens are coming: when that is more than the split was made for, the pool works
        out the memory again (before the first step when the Trainer says its batch size, so the devices load their layers once)."""
        shape = (rows, -(-int(seq) // 128) * 128)
        seen = getattr(self, "_fitted", (0, 0))
        told = getattr(self, "_pieces", None)
        if shape[0] > seen[0] or shape[1] > seen[1] or (pieces and pieces != told):
            self._fitted = (max(shape[0], seen[0]), max(shape[1], seen[1]))
            self._pieces = pieces or told
            fit = getattr(self.pool, "fit_batch", None)
            if fit is not None:
                fit(*self._fitted, pieces=self._pieces)

    def _messages(self, box, hidden, micro_rows=None):
        rows = hidden.shape[0]
        cuts = _split_points(rows, self.job.get("micro", 8), micro_rows or self.job.get("micro_batch_size"))
        if hidden.dim() > 2 and hidden.shape[1] == rows:
            cuts = [(0, rows)]
        trees, origins = split_args(box.per_block, rows, cuts)
        messages, sides = [], []
        for (a, b), tree, origin in zip(cuts, trees, origins):
            meta, arrays, micro_sides = pack_args(tree, carried=box.carried)
            sides.append([origin.get(id(s), (s, None, a, b)) for s in micro_sides])
            h = hidden[a:b].detach().float().cpu().numpy()
            messages.append(pack({"args": meta}, **with_prefix(encode(h, self.io_codec), "h_"), **with_prefix(arrays, "a_")))
        return cuts, messages, sides

    def forward(self, box, hidden):
        self.drop_pending(redo=True)
        cuts, messages, sides = self._messages(box, hidden)
        if box.train and hidden.dim() > 2:
            self.expect(max(b - a for a, b in cuts), int(hidden.shape[1]), len(cuts))
        tokens = int(hidden.shape[0] * hidden.shape[1]) if hidden.dim() > 2 else int(hidden.shape[0])
        if not box.train:
            return self._join(self.pool.forward(messages, adapter=box.adapter), hidden)
        state = {"outputs": queue.Queue(), "cots": queue.Queue(), "result": queue.Queue(), "done": threading.Event(), "cuts": cuts, "sides": sides,
                 "messages": messages, "tokens": tokens, "redo": False}
        box.state = state
        need_grads = bool(box.hidden.requires_grad or box.sides)

        def provider(outputs):
            state["outputs"].put(outputs)
            cots = state["cots"].get()
            if cots is None:
                shapes = [decode(split_prefix(unpack(o)[1], "h_")).shape for o in outputs]
                return [pack(None, **with_prefix(encode(np.zeros(s, np.float32), self.io_codec), "g_")) for s in shapes]
            return cots

        def work():
            try:
                state["result"].put(self.pool.forward_backward(messages, provider, tokens=tokens, grads=need_grads))
            except BaseException as e:
                state["outputs"].put(e)
                state["result"].put(e)
            finally:
                state["done"].set()
        threading.Thread(target=work, daemon=True).start()
        got = state["outputs"].get()
        if isinstance(got, BaseException):
            raise got
        self.pending = box
        return self._join(got, hidden)

    def _join(self, outputs, like):
        import torch
        parts = [decode(split_prefix(unpack(o)[1], "h_")) for o in outputs]
        return torch.tensor(np.concatenate(parts, 0), dtype=like.dtype, device=like.device)

    def backward(self, box, grad):
        import torch
        state = box.state
        if self.pending is box:
            self.pending = None
        g = grad.detach().float().cpu().numpy()
        cots = [pack(None, **with_prefix(encode(g[a:b], self.io_codec), "g_")) for a, b in state["cuts"]]
        if state["redo"]:
            self.drop_pending(redo=True)
            result = self.pool.forward_backward(state["messages"], lambda outputs: cots, tokens=0,
                                                grads=bool(box.hidden.requires_grad or box.sides))
        else:
            state["cots"].put(cots)
            result = state["result"].get()
            if isinstance(result, BaseException):
                raise result
        norm = float(np.sqrt(max(0.0, result.get("grad_sq", 0.0))))
        g_hidden = None
        side_grads = [None] * len(box.sides)
        if result.get("grads"):
            index = {id(s): i for i, s in enumerate(box.sides)}
            rows = []
            for m, blob in enumerate(result["grads"]):
                arrays = unpack(blob)[1]
                rows.append(decode(split_prefix(arrays, "g_")))
                for k, (orig, dim, a, b) in enumerate(state["sides"][m]):
                    if not any(name.startswith(f"s{k}_") for name in arrays):
                        continue
                    i = index.get(id(orig))
                    if i is None:
                        continue
                    piece = torch.tensor(decode(split_prefix(arrays, f"s{k}_")))
                    if side_grads[i] is None:
                        side_grads[i] = torch.zeros(orig.shape, dtype=torch.float32)
                    if dim == 0:
                        side_grads[i][a:b] += piece
                    elif dim == 1:
                        side_grads[i][:, a:b] += piece
                    else:
                        side_grads[i] += piece
            g_hidden = torch.tensor(np.concatenate(rows, 0), dtype=box.hidden.dtype)
            side_grads = [None if v is None else v.to(s.dtype) for v, s in zip(side_grads, box.sides)]
        return g_hidden, side_grads, norm


class LocalChain:
    """The device side in this process: block cores run one after another (the self-check and tests use it)."""

    def __init__(self, cores):
        self.cores = cores

    def drop_pending(self, redo: bool = False):
        pass

    def forward(self, box, hidden):
        meta, arrays, sides = pack_args(box.per_block, carried=getattr(box, "carried", None))
        plan = meta.get("carry")
        box.state = {"packed_sides": sides, "steps": [], "plan": plan}
        h, carry = hidden, {}
        import torch
        keep = torch.is_grad_enabled()
        for ci, core in enumerate(self.cores):
            args, leaves = unpack_args(meta, arrays, core.device, core.dtype, range(core.first, core.last))
            box.state["steps"].append((h, args, leaves, carry))
            kept = {"key": (id(box), ci), "leaves": leaves} if keep else {}
            if plan:
                h, carry = core.forward(h, args, carry, plan, **kept)
            else:
                h = core.forward(h, args, **kept)
        return h.to(hidden.device, hidden.dtype)

    def backward(self, box, grad):
        g = grad
        side_grads = {}
        plan, carry_grads = box.state["plan"], {}
        for ci in reversed(range(len(self.cores))):
            core, (x, args, leaves, carry) = self.cores[ci], box.state["steps"][ci]
            if plan:
                g, sides, carry_grads = core.backward(x, args, leaves, g, carry, carry_grads, plan, key=(id(box), ci))
            else:
                g, sides = core.backward(x, args, leaves, g, key=(id(box), ci))
            for k, v in sides.items():
                if v is not None:
                    side_grads[k] = side_grads.get(k, 0) + v.float().cpu()
        index = {id(s): i for i, s in enumerate(box.sides)}
        out = [None] * len(box.sides)
        for k, s in enumerate(box.state["packed_sides"]):
            if k in side_grads and id(s) in index:
                out[index[id(s)]] = side_grads[k].to(s.dtype)
        norm = math.sqrt(sum(c.grad_sq() for c in self.cores))
        return g.to("cpu", box.hidden.dtype), out, norm

import functools
import inspect
import json
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .. import fetch
from ..blocks import BETAS, EPS, SCALE_GROWTH_STEPS, lora_init, targeted
from .common import random_weights


class LoRA(nn.Module):
    def __init__(self, base, A: mx.array, B: mx.array, scale: float):
        super().__init__()
        self.base, self.scale = base, scale
        self.enabled = True
        self.A, self.B = A, B

    def __call__(self, x):
        if not self.enabled:
            return self.base(x)
        return self.base(x) + (self.scale * ((x.astype(mx.float32) @ self.A.T) @ self.B.T)).astype(x.dtype)


class Body(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers


def _mlx_lm_classes():
    try:
        from mlx_lm.utils import _get_classes
    except ImportError:
        import importlib.metadata
        try:
            version = importlib.metadata.version("mlx-lm")
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError("MLX training needs mlx-lm: run pip install mlx-lm, or start this device with --backend torch") from None
        raise RuntimeError(f"this mlx-lm ({version}) has no mlx_lm.utils._get_classes, which layerpool uses to build MLX layers; "
                           "install the mlx-lm version layerpool was released with, or start this device with --backend torch") from None
    return _get_classes


MX_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float32", "U32": "uint32", "I32": "int32", "U8": "uint8",
             "I8": "int8", "I64": "int64", "U16": "uint16", "I16": "int16", "BOOL": "bool_"}


def _stack_owner(module, n: int, path: str = ""):
    """The module holding the model's list of n blocks, and its path: mlx-lm keeps it under different names per family (model,
    language_model.model, ...)."""
    for key, value in module.items():
        if key == "layers" and isinstance(value, list) and len(value) == n:
            return module, path
    for key, value in module.items():
        if isinstance(value, nn.Module):
            found = _stack_owner(value, n, f"{path}.{key}" if path else key)
            if found:
                return found
    return None


def _placeholders(repo: str, real: dict) -> dict:
    """Every tensor of the checkpoint: the ones loaded, and zeros of the right shape for the rest. MLX builds zeros only when they
    are used, and they never are: they only let mlx-lm's sanitize see the whole checkpoint, as some families decide by it how to
    convert their weights."""
    out = {}
    for name in sorted(set(fetch.weight_map(repo).values())):
        for key, info in fetch.header(repo, name).items():
            if key == "__metadata__" or key in real:
                continue
            out[key] = mx.zeros(tuple(info["shape"]), getattr(mx, MX_DTYPES.get(info["dtype"], "float32")))
    out.update(real)
    return out


def _for_mlx_lm(config: dict) -> dict:
    """The config as mlx-lm reads it. transformers 5 writes rope settings inside rope_parameters, where many mlx-lm families do
    not look (they would fall back to their default base), and keeps a hybrid model's pattern only as layer_types, where mlx-lm
    reads full_attention_interval; a text-only config of a vision-language family (qwen3_5_text) is built by that family's code."""
    import importlib.util
    config = json.loads(json.dumps(config))
    for c in [config] + ([config["text_config"]] if isinstance(config.get("text_config"), dict) else []):
        rope = c.get("rope_parameters")
        if isinstance(rope, dict) and isinstance(rope.get("full_attention"), dict):
            local = rope.get("sliding_attention")
            if isinstance(local, dict) and c.get("rope_local_base_freq") is None and "rope_theta" in local:
                c["rope_local_base_freq"] = local["rope_theta"]
            rope = rope["full_attention"]
        if isinstance(rope, dict):
            for k in ("rope_theta", "partial_rotary_factor"):
                if c.get(k) is None and k in rope:
                    c[k] = rope[k]
            if rope.get("rope_type") not in (None, "default") and not c.get("rope_scaling"):
                c["rope_scaling"] = {**rope, "type": rope["rope_type"]}
        kinds = c.get("layer_types")
        if isinstance(kinds, list) and "full_attention" in kinds:
            every = kinds.index("full_attention") + 1
            if all((kind == "full_attention") == ((i + 1) % every == 0) for i, kind in enumerate(kinds)):
                if "linear_attention" in kinds and c.get("full_attention_interval") is None:
                    c["full_attention_interval"] = every
                if "sliding_attention" in kinds and c.get("sliding_window_pattern") is None:
                    c["sliding_window_pattern"] = every
    kind = str(config.get("model_type") or "")
    if kind.endswith("_text") and importlib.util.find_spec(f"mlx_lm.models.{kind}") is None:
        config["model_type"] = kind[:-len("_text")]
    return config


def layers_path(repo: str, n: int) -> str:
    """The path of the list of blocks in mlx-lm's model of this checkpoint (model.layers, language_model.model.layers, ...)."""
    config = _for_mlx_lm(fetch.load_json(repo, "config.json"))
    model_class, args_class = _mlx_lm_classes()(config)
    found = _stack_owner(model_class(args_class.from_dict(config)), n)
    if found is None:
        raise ValueError(f"mlx-lm's model of {repo} has no list of {n} blocks")
    return f"{found[1]}.layers" if found[1] else "layers"


def load_range(repo: str, first: int, last: int, n: int, stack: str = "model.layers", dtype=None, random_init: bool = False, log=None):
    """Blocks first..last of a model mlx-lm can run, built by mlx-lm's own code for its family, without the embeddings and the
    output head (those stay on the training machine). The whole model is described, so each block keeps what depends on its place
    (linear or full attention, sliding windows), but only these blocks are ever made in memory."""
    from ..blocks import block_spans
    import re
    config = fetch.load_json(repo, "config.json")
    model_type = str(config.get("model_type") or (config.get("text_config") or {}).get("model_type") or "")
    model_class, args_class = _mlx_lm_classes()(_for_mlx_lm(config))
    model = model_class(args_class.from_dict(_for_mlx_lm(config)))
    found = _stack_owner(model, n)
    if found is None:
        raise ValueError(f"mlx-lm's {model_type} model has no list of {n} blocks layerpool can find")
    owner, path = found
    prefix = f"{path}.layers." if path else "layers."
    layers = list(owner["layers"])[first:last]
    owner["layers"] = layers
    body = Body(layers)
    renamed = {}
    if random_init:
        local = dict(tree_flatten(body.parameters()))
        made = random_weights({f"{stack}.{int(k.split('.')[1]) + first}.{k.split('.', 2)[2]}": tuple(v.shape) for k, v in local.items()},
                              config)
        renamed = {k: mx.array(made[f"{stack}.{int(k.split('.')[1]) + first}.{k.split('.', 2)[2]}"]) for k in local}
    else:
        spans = block_spans(fetch.weight_map(repo), stack, n)
        wanted = [k for k, (i, _, _) in spans.items() if first <= i < last]
        real = {}
        for file, keys in fetch.fetch(repo, wanted, log).items():
            arrays = mx.load(file)
            real.update({k: arrays[k] for k in keys})
        weights = model.sanitize(_placeholders(repo, real)) if hasattr(model, "sanitize") else real
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.(.+)$")
        for key, value in weights.items():
            hit = pattern.match(key)
            if hit and first <= int(hit.group(1)) < last:
                renamed[f"layers.{int(hit.group(1)) - first}.{hit.group(2)}"] = value
        quant = config.get("quantization") or (config.get("text_config") or {}).get("quantization")
        if quant:
            extra = {"mode": quant.get("mode", "affine")} if "mode" in inspect.signature(nn.quantize).parameters else {}

            def chosen(p, m):
                if not hasattr(m, "to_quantized") or f"{p}.scales" not in renamed:
                    return False
                j, rest = p.split(".", 2)[1:]
                own = quant.get(f"{prefix}{int(j) + first}.{rest}")
                return own if isinstance(own, dict) else True
            nn.quantize(body, group_size=quant["group_size"], bits=quant["bits"], **extra, class_predicate=chosen)
    body.load_weights(list(renamed.items()), strict=False)
    missing = [k for k, _ in tree_flatten(body.parameters()) if k not in renamed and not k.endswith((".A", ".B"))]
    if missing:
        raise ValueError(f"{repo} has no weights for {missing[:5]}{' ...' if len(missing) > 5 else ''} of its blocks as mlx-lm builds "
                         "them; this model's layout is not supported by the MLX backend")
    if dtype is not None:
        body.set_dtype(dtype)
    body.train()
    del model
    return body, layers


def _pinned(fn):
    @functools.wraps(fn)
    def run(self, *args, **kwargs):
        return self.thread.submit(fn, self, *args, **kwargs).result()
    return run



def _linear_names(layer) -> list[str]:
    """Every linear layer in a block, sorted by name: the same LoRA targets and order as the torch blocks."""
    return sorted(name for name, mod in layer.named_modules()
                  if name and not name.endswith(".base") and isinstance(mod, (nn.Linear, nn.QuantizedLinear, LoRA)))


def _get(module, path: str):
    for part in path.split(".") if path else []:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


class StageCore:
    """Consecutive blocks on one Apple GPU (or CPU) with MLX: forward, backward, its own AdamW, and its state for checkpoints,
    the same as the torch BlockCore."""

    def __init__(self, loader, first: int, rank: int, scale: float, lr: float, loss_scale: float = 1.0, checkpointing: bool = True,
                 device=None, mode: str = "lora", weight_decay: float = 0.0, targets: list[str] | None = None):
        device = device or mx.default_device()
        self.targets = targets
        self.thread = ThreadPoolExecutor(1, initializer=mx.set_default_device, initargs=(device,))
        self._setup(loader, first, rank, scale, lr, loss_scale, checkpointing, mode, weight_decay)

    @_pinned
    def _setup(self, loader, first, rank, scale, lr, loss_scale, checkpointing, mode, weight_decay):
        model, layers = loader()
        mx.eval(model.parameters())
        self.model, self.first, self.mode = model, first, mode
        self.loss_scale, self.checkpointing, self.lr, self.weight_decay = loss_scale, checkpointing, lr, weight_decay
        self.max_loss_scale, self.good_steps, self.overflows = max(1.0, loss_scale), 0, 0
        self.body = Body(layers)
        self.layers = self.body.layers
        self.model.freeze()
        self.init, self.loras, names = {}, [], []
        if mode == "full":
            self.body.unfreeze()
            for k, _ in tree_flatten(self.body.trainable_parameters()):
                j, rest = k.split(".", 2)[1:]
                names.append((k, f"{first + int(j)}.{rest}"))
        else:
            for layer in self.layers:
                for name, module in [(n, m) for n, m in layer.named_modules() if isinstance(m, LoRA)]:
                    parent, _, attr = name.rpartition(".")
                    setattr(_get(layer, parent), attr, module.base)
            for j, layer in enumerate(self.layers):
                for k, name in enumerate(_linear_names(layer)):
                    if not targeted(name, self.targets):
                        continue
                    parent_name, _, attr = name.rpartition(".")
                    holder = _get(layer, parent_name)
                    base = getattr(holder, attr)
                    base = base.base if isinstance(base, LoRA) else base
                    fan_out = base.weight.shape[0]
                    fan_in = base.weight.shape[1] * 32 // base.bits if isinstance(base, nn.QuantizedLinear) else base.weight.shape[1]
                    A, B = lora_init(first + j, k, fan_in, fan_out, rank)
                    lora = LoRA(base, mx.array(A), mx.array(B), scale)
                    lora.freeze()
                    lora.unfreeze(keys=["A", "B"], recurse=False)
                    setattr(holder, attr, lora)
                    self.loras.append(lora)
                    self.init[f"{first + j}.{name}"] = (first + j, k, fan_in, fan_out, rank)
                    names.append((f"layers.{j}.{name}.A", f"{first + j}.{name}.A"))
                    names.append((f"layers.{j}.{name}.B", f"{first + j}.{name}.B"))
        if mode != "full" and not self.loras:
            raise ValueError(f"target_modules {self.targets} names none of the linear layers of the MLX blocks")
        self.names = dict(names)
        self.dtype = self._dtype()
        if self.dtype == mx.float32:
            self.loss_scale = self.max_loss_scale = 1.0
        self.grads = None
        self.graphs = {}
        self.m = self.v = None
        self.t = 0
        self.params = {ours: None for ours in self.names.values()}
        self.initial_state = self._flat_params() if mode == "full" else None
        if self.initial_state is not None:
            self.initial_state = {self.names[k]: np.array(v.astype(mx.float32)) for k, v in self.initial_state.items()}

    def _flat_params(self) -> dict:
        return dict(tree_flatten(self.body.trainable_parameters()))

    def _update(self, flat: dict):
        if flat:
            self.body.update(tree_unflatten(list(flat.items())))

    def _dtype(self):
        for _, value in tree_flatten(self.layers[0].parameters()):
            if mx.issubdtype(value.dtype, mx.floating) and value.dtype != mx.float32:
                return value.dtype
        return mx.float32

    @_pinned
    def tensor(self, array: np.ndarray):
        return mx.array(array.astype(np.int32) if np.issubdtype(array.dtype, np.integer) else array)

    @_pinned
    def numpy(self, value) -> np.ndarray:
        return np.array(value.astype(mx.float32))

    @_pinned
    def set_adapter(self, enabled: bool):
        for lora in self.loras:
            lora.enabled = enabled

    def _layer(self, layer, h, mask):
        if not self.checkpointing:
            return layer(h, mask, None)

        def inner(params, h):
            layer.update(params)
            return layer(h, mask, None)
        return mx.checkpoint(inner)(layer.trainable_parameters(), h)

    def _mask(self, attend, made, linear: bool):
        """The mask one block takes, from where each token may look (rows, 1, tokens, tokens), or from which tokens are real
        (rows, tokens): full-attention blocks take the first, linear-attention blocks (mlx-lm marks them is_linear) the second.
        None means plainly causal with no padding."""
        if attend is None:
            return None if linear else "causal"
        key = (id(attend), linear)
        if key not in made:
            if linear:
                keep = attend if attend.ndim == 2 else attend[:, 0].any(axis=-2)
                made[key] = None if keep.all() else mx.array(keep)
            else:
                if attend.ndim == 2:
                    q = attend.shape[1]
                    attend = np.tril(np.ones((q, q), bool))[None, None] & attend[:, None, None, :]
                attend = attend | ~attend.any(-1, keepdims=True)
                q, k = attend.shape[-2:]
                causal = np.tril(np.ones((q, k), bool), k - q)
                made[key] = "causal" if (attend == causal).all() else mx.array(attend)
        return made[key]

    def _run(self, h, masks=None):
        """masks: {block index: where each token may look (rows, 1, tokens, tokens), or which tokens are real (rows, tokens)},
        from the masks transformers gives its blocks, so padded rows train as there; a block without one is causal."""
        made = {}
        for j, layer in enumerate(self.layers):
            attend = (masks or {}).get(self.first + j)
            mask = self._mask(attend, made, bool(getattr(layer, "is_linear", False))) if h.shape[1] > 1 else None
            h = self._layer(layer, h, mask)
        return h

    @_pinned
    def forward(self, x, args=None, key=None, leaves=None):
        """The slice's output. With a key (a micro-batch whose backward comes later), each block's input is kept, so the
        backward runs every block forward once more instead of the whole slice twice."""
        if key is None:
            out = self._run(x.astype(self.dtype), args)
            mx.eval(out)
            return out
        h, made, kept = x.astype(self.dtype), {}, []
        for j, layer in enumerate(self.layers):
            attend = (args or {}).get(self.first + j)
            mask = self._mask(attend, made, bool(getattr(layer, "is_linear", False))) if h.shape[1] > 1 else None
            kept.append((h, mask))
            h = layer(h, mask, None)
        mx.eval(h, *[k[0] for k in kept])
        self.graphs[key] = kept
        return h

    @_pinned
    def backward(self, x, args=None, leaves=None, grad_out=None, key=None):
        """Blocks only: the gradient of the input for the given output gradient (unscaled in and out, like the torch blocks)."""
        kept = self.graphs.pop(key, None) if key is not None else None
        if kept is None and x is None:
            raise RuntimeError("layerpool: a backward came for a micro-batch whose forward is gone")
        while True:
            grad_cot = (grad_out.astype(mx.float32) * self.loss_scale).astype(self.dtype)
            if kept is not None:
                grads = {}
                for j in reversed(range(len(self.layers))):
                    layer, (h, mask) = self.layers[j], kept[j]

                    def objective(params, inp, layer=layer, mask=mask, cot=grad_cot):
                        layer.update(params)
                        return (layer(inp, mask, None) * cot).astype(mx.float32).sum()
                    got, grad_cot = mx.grad(objective, argnums=(0, 1))(layer.trainable_parameters(), h)
                    for k, v in tree_flatten(got):
                        grads[f"layers.{j}.{k}"] = v
                    mx.eval(grad_cot, list(grads.values()))
                grad_in = grad_cot
            else:
                def whole(params, inp, cot=grad_cot):
                    self._update(params)
                    return (self._run(inp, args) * cot).astype(mx.float32).sum()
                grads, grad_in = mx.grad(whole, argnums=(0, 1))(self._flat_params(), x.astype(self.dtype))
                mx.eval(grads, grad_in)
            finite = bool(mx.all(mx.array([mx.isfinite(v).all() for v in [grad_in, *grads.values()]])).item())
            if finite or self.loss_scale <= 1.0:
                break
            self.loss_scale, self.good_steps = self.loss_scale / 2, 0
            self.overflows += 1
        grads = {k: v.astype(mx.float32) / self.loss_scale for k, v in grads.items()}
        self.grads = grads if self.grads is None else {k: self.grads[k] + grads[k] for k in grads}
        out = grad_in.astype(mx.float32) / self.loss_scale
        mx.eval(self.grads, out)
        return out, {}

    @_pinned
    def grad_sq(self) -> float:
        if self.grads is None:
            return 0.0
        return float(sum((v.astype(mx.float32) ** 2).sum() for v in self.grads.values()).item())

    @_pinned
    def step(self, lr: float | None = None, clip: float = 1.0, skip: bool = False):
        if self.grads is None or skip:
            self.grads = None
            return
        if lr is not None:
            self.lr = lr
        params = self._flat_params()
        grads = {k: v.astype(mx.float32) * clip for k, v in self.grads.items()}
        if self.m is None:
            self.m = {k: mx.zeros(v.shape, mx.float32) for k, v in params.items()}
            self.v = {k: mx.zeros(v.shape, mx.float32) for k, v in params.items()}
        self.t += 1
        b1, b2 = BETAS
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        new = {}
        for k, p in params.items():
            g = grads[k]
            p32 = p.astype(mx.float32)
            if self.weight_decay and p.ndim > 1:
                p32 = p32 * (1 - self.lr * self.weight_decay)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            new[k] = (p32 - (self.lr / c1) * self.m[k] / (mx.sqrt(self.v[k]) / (c2 ** 0.5) + EPS)).astype(p.dtype)
        self._update(new)
        mx.eval(list(new.values()), list(self.m.values()), list(self.v.values()))
        self.grads = None
        self.good_steps += 1
        if self.good_steps >= SCALE_GROWTH_STEPS and self.loss_scale < self.max_loss_scale:
            self.loss_scale, self.good_steps = min(self.max_loss_scale, self.loss_scale * 2), 0

    @_pinned
    def zero_grad(self):
        self.grads = None
        self.graphs.clear()

    def set_weight_decay(self, value: float):
        self.weight_decay = value

    @_pinned
    def state(self) -> dict[str, np.ndarray]:
        params = self._flat_params()
        out = {}
        for k, ours in self.names.items():
            out[ours] = np.array(params[k].astype(mx.float32))
            if self.m is not None:
                out[ours + ".m"] = np.array(self.m[k])
                out[ours + ".v"] = np.array(self.v[k])
                out[ours + ".t"] = np.array([float(self.t)], np.float32)
        return out

    def initial(self) -> dict[str, np.ndarray]:
        if self.mode == "full":
            return dict(self.initial_state)
        out = {}
        for key, args in self.init.items():
            out[key + ".A"], out[key + ".B"] = lora_init(*args)
        return out

    @_pinned
    def load_state(self, saved: dict[str, np.ndarray]) -> int:
        params = self._flat_params()
        new, restored = {}, 0
        if not any(k.endswith(".m") for k in saved):
            self.m = self.v = None
            self.t = 0
        for k, ours in self.names.items():
            if ours in saved:
                new[k] = mx.array(saved[ours].astype(np.float32)).astype(params[k].dtype)
                restored += 1
                if ours + ".m" in saved:
                    if self.m is None:
                        self.m = {key: mx.zeros(value.shape, mx.float32) for key, value in params.items()}
                        self.v = {key: mx.zeros(value.shape, mx.float32) for key, value in params.items()}
                    self.m[k] = mx.array(saved[ours + ".m"].astype(np.float32))
                    self.v[k] = mx.array(saved[ours + ".v"].astype(np.float32))
                    self.t = int(saved[ours + ".t"][0])
        if new:
            self._update(new)
            mx.eval(list(new.values()))
        return restored


def verify(repo: str, auto: str, mode: str, log=print) -> str | None:
    """Before MLX trains blocks of an architecture, a small copy of it is built and the same blocks are run by transformers (PyTorch)
    and by mlx-lm (MLX) on a left-padded batch: outputs, the gradient going back and the trained weights' gradients must match,
    and in full fine-tuning the weights themselves. Returns None when they do, else why not. Each architecture is checked once per
    mlx-lm, transformers and layerpool version."""
    import importlib.metadata
    import tempfile

    import torch
    import transformers
    from transformers.utils import logging

    from .. import __version__
    from ..blocks import BlockCore, layout, load_blocks, load_config, mlx_masks, pack_args, unpack_args
    from ..job import load_config as raw_config
    from ..selfcheck import _remember, _remembered, _small, kept_rng, settings_hash
    config = load_config(repo)
    kind = str(getattr(config.get_text_config(), "model_type", "") or getattr(config, "model_type", ""))
    key = (f"mlx|{kind}|{settings_hash(config)}|{auto}|{mode}|mlx-lm {importlib.metadata.version('mlx-lm')}"
           f"|transformers {transformers.__version__}|layerpool {__version__}")
    known = _remembered(key)
    if known is not None:
        return None if known == "ok" else known
    level = logging.get_verbosity()
    logging.set_verbosity_error()
    try:
        with tempfile.TemporaryDirectory(prefix="layerpool_mlx_check_") as folder:
            reason = kept_rng(_compare)(config, raw_config(repo), auto, mode, folder, torch, transformers, BlockCore, layout, load_blocks,
                                        load_config, mlx_masks, pack_args, unpack_args, _small)
    except Exception as e:
        return f"the check could not run ({type(e).__name__}: {str(e)[:200]})"
    finally:
        logging.set_verbosity(level)
    _remember(key, reason or "ok")
    return reason


def _compare(config, raw, auto, mode, folder, torch, transformers, BlockCore, layout, load_blocks, load_config, mlx_masks, pack_args,
             unpack_args, _small) -> str | None:
    small = type(config).from_dict(_small(config.to_dict()))
    torch.manual_seed(0)
    whole = getattr(transformers, auto).from_config(small).float().eval()
    with torch.no_grad():
        for name, p in whole.named_parameters():
            if "norm" in name and p.is_floating_point():
                p.add_(torch.randn_like(p) * 0.3)
    whole.save_pretrained(folder)
    import os
    import shutil
    theirs_folder = os.path.join(folder, "as_run")
    os.makedirs(theirs_folder)
    for name in os.listdir(folder):
        if name.endswith((".safetensors", ".json")) and name != "config.json":
            shutil.copy(os.path.join(folder, name), theirs_folder)
    with open(os.path.join(theirs_folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_small(raw), f)
    lay = layout(load_config(folder), auto)
    n = lay["n"]
    first, last = (1, n) if n > 2 else (0, n)
    blocks = whole.get_submodule(lay["stack"])
    vocab = min(getattr(small.get_text_config(), "vocab_size", 300) or 300, 300)
    ids = torch.randint(3, vocab, (3, 64), generator=torch.Generator().manual_seed(1))
    keep = torch.ones(3, 64, dtype=torch.long)
    keep[0, :5] = 0
    keep[1, :2] = 0
    seen = {}

    def grab(i):
        def hook(module, args, kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in ("past_key_values", "past_key_value", "cache_params")}
            if args:
                seen[i] = (tuple(args[1:]), kwargs)
                hidden = args[0]
            else:
                kwargs = dict(kwargs)
                hidden = kwargs.pop("hidden_states")
                seen[i] = ((), kwargs)
            if i == first:
                seen["input"] = hidden.detach()
        return hook
    hooks = [blocks[i].register_forward_pre_hook(grab(i), with_kwargs=True) for i in range(first, last)]
    with torch.no_grad():
        whole(input_ids=ids, attention_mask=keep, use_cache=False)
    for h in hooks:
        h.remove()
    meta, arrays, _ = pack_args({i: seen[i] for i in range(first, last)}, grads=False)
    masks = mlx_masks(meta, arrays, range(first, last))
    args, leaves = unpack_args(meta, arrays, "cpu", torch.float32, range(first, last))
    ours = BlockCore(load_blocks(folder, lay, first, last, "cpu", torch.float32), first, 4, 2.0, 1e-2, loss_scale=1.0, mode=mode)
    theirs = StageCore(lambda: load_range(theirs_folder, first, last, n, lay["stack"], mx.float32), first, 4, 2.0, 1e-2, loss_scale=1.0,
                       mode=mode, device=mx.cpu)
    x = seen["input"]
    real = keep.numpy().astype(bool)

    def close(a, b, what):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        if a.shape != b.shape:
            return f"{what}: shapes {a.shape} and {b.shape} differ"
        gap, scale = float(np.abs(a - b).max()) if a.size else 0.0, float(np.abs(a).max()) if a.size else 0.0
        return None if gap <= 5e-4 * max(1.0, scale) else f"{what} differ by {gap:.3g} (largest value {scale:.3g})"
    if mode == "full":
        mine = {k: v for k, v in ours.state().items() if not k.endswith((".m", ".v", ".t"))}
        other = {k: v for k, v in theirs.state().items() if not k.endswith((".m", ".v", ".t"))}
        if set(mine) != set(other):
            return f"the blocks' weights are named differently ({sorted(set(mine) ^ set(other))[:3]})"
        for k in mine:
            wrong = close(mine[k], other[k], f"weight {k}")
            if wrong:
                return wrong
    out = ours.forward(x, args)
    got = np.array(theirs.forward(theirs.tensor(x.numpy()), masks, key=0))
    wrong = close(out.detach().numpy()[real], got[real], "outputs")
    if wrong:
        return wrong
    g = np.random.default_rng(0).standard_normal(out.shape).astype(np.float32) * real[..., None]
    g_ours, _ = ours.backward(x, args, leaves, torch.from_numpy(g))
    g_theirs, _ = theirs.backward(None, None, None, theirs.tensor(g), key=0)
    wrong = close(g_ours.numpy()[real], np.array(g_theirs)[real], "input gradients")
    if wrong:
        return wrong
    mine = {k: p.grad.detach().numpy() for k, p in ours.params.items() if p.grad is not None}
    other = {theirs.names[k]: np.array(v.astype(mx.float32)) for k, v in tree_flatten(theirs.grads)}
    if set(mine) != set(other):
        return f"the trained weights are named differently ({sorted(set(mine) ^ set(other))[:3]})"
    for k in mine:
        wrong = close(mine[k], other[k], f"gradient of {k}")
        if wrong:
            return wrong
    if mx.default_device() != mx.cpu:
        try:
            gpu = StageCore(lambda: load_range(theirs_folder, first, last, n, lay["stack"], mx.float32), first, 4, 2.0, 1e-2, loss_scale=1.0,
                            mode=mode, device=mx.default_device())
            got_gpu = np.array(gpu.forward(gpu.tensor(x.numpy()), masks, key=0))
            g_gpu, _ = gpu.backward(None, None, None, gpu.tensor(g), key=0)
        except Exception as e:
            return f"the blocks fail on the Apple GPU ({type(e).__name__}: {str(e)[:150]})"
        for a, b, what in ((out.detach().numpy()[real], got_gpu[real], "outputs on the Apple GPU"),
                           (g_ours.numpy()[real], np.array(g_gpu)[real], "input gradients on the Apple GPU")):
            gap, scale = float(np.abs(a - b).max()), float(np.abs(a).max())
            if gap > 2e-2 * max(1.0, scale):
                return f"{what} differ by {gap:.3g} (largest value {scale:.3g})"
    return None

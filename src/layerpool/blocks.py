"""Any transformers model, split by its repeated blocks.

A model is a stack of identical blocks with some modules around it (embeddings, an image or audio encoder, the output head).
The machine that runs the training code keeps the modules around the stack and runs the model's own forward from transformers;
the devices hold slices of the stack. Each block is called with the arguments the model itself computed for it (masks, rotary
tables, positions, encoder states), so no architecture needs code of its own here."""
import copy
import hashlib
import json
import os
import re
import shutil
import time

import numpy as np

from . import fetch
from .backends import common

STACK_COUNT_ATTRS = ("num_hidden_layers", "n_layer", "num_layers", "decoder_layers", "n_layers", "num_decoder_layers")
RANGE_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "layerpool", "ranges")
BETAS = (0.9, 0.999)
SCALE_GROWTH_STEPS = 200
EPS = 1e-8


def auto_class(name: str):
    import transformers
    return getattr(transformers, name)


def stack_candidates(model) -> list[tuple[str, object]]:
    """Lists of same-class modules that could be the repeated blocks, most likely first: those as long as a layer count of the
    model's config (the text model's first), then the biggest. Lists of embeddings, convolutions or norms are not blocks."""
    import torch
    counts = set()
    for config in (getattr(model, "config", None), getattr(getattr(model, "config", None), "get_text_config", lambda: None)()):
        for attr in STACK_COUNT_ATTRS:
            value = getattr(config, attr, None) if config is not None else None
            if isinstance(value, int):
                counts.add(value)
    leaf = (torch.nn.Embedding, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.LayerNorm, torch.nn.Linear)
    found = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 1 and len({type(m) for m in mod}) == 1 and not isinstance(mod[0], leaf):
            size = sum(p.numel() for p in mod.parameters())
            found.append((len(mod) in counts, size, name, mod))
    found.sort(key=lambda f: (f[0], f[1]), reverse=True)
    return [(name, mod) for _, _, name, mod in found]


def _parallel_lists(model, path: str, stack) -> list[str]:
    """Other lists as long as the stack beside it (XLM keeps attentions, norms and FFNs as parallel lists): such a model's
    layers are not single modules, and splitting one of the lists would compute something else."""
    import torch
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    return [n for n, m in parent.named_children() if n != name and isinstance(m, torch.nn.ModuleList) and len(m) == len(stack)]


def load_config(repo: str):
    from transformers import AutoConfig
    check_source(repo)
    return AutoConfig.from_pretrained(repo)


def check_source(repo: str):
    """A model given as a folder must exist on this device too; say so instead of a Hugging Face "not found"."""
    looks_local = os.path.isabs(repo) or repo.startswith((".", "~")) or "\\" in repo or repo.count("/") > 1
    if looks_local and not os.path.isdir(os.path.expanduser(repo)):
        raise FileNotFoundError(f"the model folder {repo} is not on this device ({os.uname().nodename if hasattr(os, 'uname') else 'here'}): "
                                "copy it to the same path on every device, or push it to the Hugging Face Hub and use its name")


def skeleton(config, auto: str):
    import torch
    cls = auto_class(auto)
    try:
        cls._model_mapping[type(config)]
    except Exception:
        pass
    with torch.device("meta"):
        return cls.from_config(copy.deepcopy(config))


def _sub(config, sub):
    return config if sub is None else getattr(config, sub)


def _subconfigs(config) -> list:
    names = [None]
    for key in sorted(vars(config)):
        try:
            value = getattr(config, key, None)
        except Exception:
            value = vars(config).get(key)
        if key.endswith("_config") and hasattr(value, "to_dict") and value is not config:
            names.append(key)
    return names


def with_count(config, sub, attr: str, first: int, last: int):
    """A copy of `config` whose stack has blocks first..last-1 of the full model, with every per-block list sliced to match."""
    return with_blocks(config, sub, attr, list(range(first, last)))


def with_blocks(config, sub, attr: str, indices: list[int]):
    """A copy of `config` whose stack has only the given blocks of the full model, with every per-block list cut to match."""
    data = config.to_dict()
    target = data if sub is None else data[sub]
    holder = _sub(config, sub)
    n = getattr(holder, attr)
    names = {attr}
    for alias, real in getattr(holder, "attribute_map", {}).items():
        if attr in (alias, real):
            names.update((alias, real))
    types = holder.__dict__.get("layer_types")
    if types is not None and len(types) == n and "layer_types" not in target:
        target["layer_types"] = list(types)
    for key, value in list(target.items()):
        if isinstance(value, list) and len(value) == n and key not in names:
            target[key] = [value[i] for i in indices]
    per_layer = target.get("per_layer_config")
    if isinstance(per_layer, dict):
        target["per_layer_config"] = {new: per_layer[k] for new, i in enumerate(indices) for k in (i, str(i)) if k in per_layer}
    for key in ("out_features", "out_indices", "stage_names"):
        target.pop(key, None)
    for name in names:
        if name in target:
            target[name] = len(indices)
    try:
        return type(config).from_dict(data)
    except Exception:
        made = copy.deepcopy(config)
        holder = _sub(made, sub)
        for key, value in target.items():
            if key in names or key == "per_layer_config" or (isinstance(value, list) and isinstance(getattr(holder, key, None), list)):
                object.__setattr__(holder, key, value)
        return made


def _signatures(holder, attr) -> list[str]:
    """For every block, the values of every per-block list in the config: blocks with the same signature are built alike."""
    n = getattr(holder, attr)
    lists = [v for v in holder.__dict__.values() if isinstance(v, list) and len(v) == n]
    return [json.dumps([v[i] for v in lists], sort_keys=True, default=str) for i in range(n)]


def _representatives(config, sub, attr) -> list[int]:
    """One block of every kind (each distinct value of every per-block list), so modules built from those lists exist."""
    holder = _sub(config, sub)
    n = getattr(holder, attr)
    picked = {0}
    lists = [v for v in holder.__dict__.values() if isinstance(v, list) and len(v) == n]
    for values in lists:
        seen = {}
        for i, v in enumerate(values):
            seen.setdefault(json.dumps(v, sort_keys=True, default=str), i)
        picked.update(seen.values())
    return sorted(picked)


NOT_SPLITTABLE = {
    "funnel": "it pools the sequence between groups of blocks",
    "fsmt": "its split outputs differ slightly from the whole model's",
    "diffusion_gemma": "its split outputs differ from the whole model's",
    "nemotron_asr_streaming_encoder": "its split outputs differ from the whole model's",
}


def layout(config, auto: str) -> dict:
    """Where the blocks are: the stack's module path, how many there are, and which config field sets that number."""
    kind = str(getattr(config, "model_type", ""))
    if kind in NOT_SPLITTABLE:
        raise ValueError(f"layerpool cannot split {kind} models yet: {NOT_SPLITTABLE[kind]}")
    model = skeleton(config, auto)
    candidates = stack_candidates(model)
    if not candidates:
        raise ValueError(f"{type(model).__name__} has no stack of repeated blocks that layerpool could split (convolutional and "
                         "staged vision models, for example, are not built that way)")
    tried = []
    for path, stack in candidates:
        n = len(stack)
        beside = _parallel_lists(model, path, stack)
        if beside:
            tried.append(f"{path} has parallel lists beside it ({', '.join(beside)})")
            continue
        for sub in _subconfigs(config):
            holder = _sub(config, sub)
            for attr in STACK_COUNT_ATTRS:
                if getattr(holder, attr, None) != n:
                    continue
                try:
                    trial = skeleton(with_count(config, sub, attr, 0, 1), auto)
                    if len(trial.get_submodule(path)) == 1:
                        return {"auto": auto, "stack": path, "n": n, "sub": sub, "attr": attr, "block": type(stack[0]).__name__}
                except Exception as e:
                    tried.append(f"{path} with {attr}=1: {type(e).__name__}: {str(e)[:120]}")
        tried.append(f"{path}: no setting of the config sets its {n} blocks")
    raise ValueError(f"layerpool cannot split {type(model).__name__}: " + "; ".join(tried[:4]))


def block_index(key: str, stack: str, strict: bool = True) -> tuple[int, int, int] | None:
    """(block index, start, end of the index in the key) when `key` belongs to a block of `stack`."""
    last = stack.split(".")[-1]
    need = set(stack.split(".")) - {"model", last}
    for found in re.finditer(rf"(?:^|\.){re.escape(last)}\.(\d+)\.", key):
        before = set(key[:found.start()].split("."))
        if strict and need - before:
            continue
        return int(found.group(1)), found.start(1), found.end(1)
    return None


def block_spans(keys, stack: str, n: int) -> dict[str, tuple[int, int, int]]:
    """Checkpoint keys of the stack's blocks: {key: (block index, start, end of the index in the key)}. Keys are grouped by what
    comes before the block number, and a group must cover every block: another list with the same name (an audio encoder's own
    "layers") is a different group. A checkpoint that stores the blocks under older names than the model's (ViT's
    vit.encoder.layer.N for vit.layers.N) is matched by its numbered group alone; transformers renames such keys when it loads."""
    keys = list(keys)
    for mode in ("strict", "loose", "any"):
        groups: dict[str, dict] = {}
        for key in keys:
            if mode == "any":
                hits = [(int(m.group(1)), m.start(1), m.end(1)) for m in re.finditer(r"\.(\d+)\.", key)]
                hit = hits[0] if hits else None
            else:
                hit = block_index(key, stack, mode == "strict")
            if hit and hit[0] < n:
                groups.setdefault(key[:hit[1]], {})[key] = hit
        complete = {p: g for p, g in groups.items() if len({h[0] for h in g.values()}) == n}
        if complete:
            words = set(re.split(r"[._]", stack))
            best = max(complete, key=lambda p: (len(words & set(re.split(r"[._]", p))), len(complete[p])))
            return complete[best]
    raise ValueError(f"the checkpoint has no weights for all {n} blocks of {stack}")


def block_keys(keys, stack: str, n: int) -> dict[str, int]:
    """Checkpoint keys of the stack's blocks, with their block index."""
    return {key: hit[0] for key, hit in block_spans(keys, stack, n).items()}


def _revision(repo: str) -> str:
    try:
        return fetch.revision(repo)
    except Exception:
        return ""


CACHE_FORMAT = 1


def _range_dir(repo: str, tag: str) -> str:
    """Where a cut-out part of a model is cached: one folder per checkpoint revision, part, cache layout and transformers
    major.minor (the version writes the config the part loads with)."""
    import transformers
    source = os.path.abspath(repo) if os.path.isdir(repo) else repo
    made_by = ".".join(transformers.__version__.split(".")[:2])
    key = hashlib.sha1(f"{source}@{_revision(repo)}|{tag}|v{CACHE_FORMAT}|tf{made_by}".encode()).hexdigest()[:16]
    if os.path.isdir(repo) and os.path.basename(source.rstrip("/")).startswith(("layerpool_check_", "layerpool_mlx_check_")):
        return os.path.join(source, ".parts", key)
    return os.path.join(RANGE_CACHE, f"{os.path.basename(source.rstrip('/'))}_{key}")


UNUSED_AFTER_S = 600


def _prune_ranges(repo: str, keep: str):
    """A device whose layers changed (a rebalance) no longer needs the folders of its earlier layer ranges of this model: they go
    once nothing has used them for ten minutes (another device on this machine may still be loading one). They are rebuilt from
    the downloaded weights if needed again."""
    source = os.path.abspath(repo) if os.path.isdir(repo) else repo
    marker = {"repo": source, "kind": "blocks"}
    os.makedirs(keep, exist_ok=True)
    with open(os.path.join(keep, "layerpool_part.json"), "w", encoding="utf-8") as f:
        json.dump(marker, f)
    now = time.time()
    for name in os.listdir(RANGE_CACHE) if os.path.isdir(RANGE_CACHE) else ():
        folder = os.path.join(RANGE_CACHE, name)
        if folder == keep:
            continue
        try:
            with open(os.path.join(folder, "layerpool_part.json"), encoding="utf-8") as f:
                theirs = json.load(f)
            if theirs == marker and now - os.path.getmtime(os.path.join(folder, "layerpool_part.json")) > UNUSED_AFTER_S:
                shutil.rmtree(folder, ignore_errors=True)
        except (OSError, ValueError):
            continue


def _write_checkpoint(out: str, config, files: dict[str, list[str]], rename):
    """A model folder with these tensors, renamed, copied byte for byte (never all in memory at once)."""
    os.makedirs(out, exist_ok=True)
    config.save_pretrained(out)
    fetch.copy_tensors(files, rename, os.path.join(out, "model.safetensors"))


def _no_quant(config):
    data = config.to_dict()
    data.pop("quantization_config", None)
    return type(config).from_dict(data)


def _overridden(config, overrides: dict | None):
    """A copy of the config with the run's settings (such as num_labels) applied."""
    config = copy.deepcopy(config)
    for key, value in (overrides or {}).items():
        setattr(config, key, value)
    return config


def _shrink_vocab(config):
    """A part with only blocks needs no real vocabulary; a small one saves memory. Configs that check the vocabulary against
    another setting keep theirs."""
    trial = copy.deepcopy(config)
    try:
        _small_vocab(trial)
        trial.to_dict()
        type(trial).from_dict(trial.to_dict())
    except Exception:
        return config
    return _small_vocab(config)


def _small_vocab(config):
    text = config.get_text_config()
    for key in ("vocab_size",):
        if getattr(text, key, None):
            setattr(text, key, 16)
    for key in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if getattr(text, key, None) is not None:
            setattr(text, key, 0)
    return config


def _load(auto: str, path: str, dtype, device, quantized: bool, overrides: dict | None = None):
    from transformers.utils import logging
    direct = quantized or not str(device).startswith("mps")
    kwargs = {**(overrides or {}), "dtype": dtype, "output_loading_info": True}
    if direct:
        kwargs["device_map"] = {"": str(device)}
    level = logging.get_verbosity()
    logging.set_verbosity_error()
    try:
        model, info = auto_class(auto).from_pretrained(path, **kwargs)
    finally:
        logging.set_verbosity(level)
    if not direct:
        model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval(), info


def random_weights(shapes: dict[str, tuple], config) -> dict:
    """Initial values for training from scratch that depend only on each weight's full-model name, so every split of the model
    starts from the same weights, on PyTorch and on MLX (backends.common.random_weights makes them for both)."""
    import torch
    text = config.get_text_config()
    made = common.random_weights(shapes, {"initializer_range": getattr(text, "initializer_range", None), "model_type": getattr(text, "model_type", "")})
    return {name: torch.from_numpy(value) for name, value in made.items()}


def load_blocks(repo: str, lay: dict, first: int, last: int, device, dtype, quantized: bool = False, random_init: bool = False, log=None,
                overrides: dict | None = None, in_place: bool = True):
    """Blocks first..last-1 of the model, built by transformers from only their own weights."""
    import torch
    stored = load_config(repo)
    if not quantized:
        stored = _no_quant(stored)
    config = _overridden(stored, overrides)
    part = _shrink_vocab(with_count(config, lay["sub"], lay["attr"], first, last))
    if random_init:
        model = auto_class(lay["auto"]).from_config(part).to(dtype)
        stack = model.get_submodule(lay["stack"])
        shapes = {f"{lay['stack']}.{first + j}.{n}": tuple(p.shape) for j, block in enumerate(stack) for n, p in block.named_parameters()}
        made = random_weights(shapes, config)
        with torch.no_grad():
            for j, block in enumerate(stack):
                for n, p in block.named_parameters():
                    p.copy_(made[f"{lay['stack']}.{first + j}.{n}"].to(p.dtype))
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        out = _range_dir(repo, f"blocks {first}-{last} q{int(quantized)}")
        _prune_ranges(repo, out)
        if not os.path.exists(os.path.join(out, "model.safetensors")):
            spans = {k: h for k, h in block_spans(fetch.weight_map(repo), lay["stack"], lay["n"]).items() if first <= h[0] < last}
            renumbered = lambda k: k[:spans[k][1]] + str(spans[k][0] - first) + k[spans[k][2]:]
            _write_checkpoint(out, _shrink_vocab(with_count(stored, lay["sub"], lay["attr"], first, last)), fetch.fetch(repo, list(spans), log), renumbered)
        model, info = _load(lay["auto"], out, dtype, device, quantized, overrides)
        prefix = lay["stack"] + "."
        missing = {k for k in info.get("missing_keys", []) if k.startswith(prefix)}
    blocks = list(model.get_submodule(lay["stack"]))
    if quantized:
        _compute_in(blocks, dtype)
    if not quantized and in_place:
        saved = None if random_init else os.path.join(out, "model.safetensors")
        blocks = _as_in_full_model(blocks, config, lay, first, device, missing if not random_init else set(), repo, saved, dtype)
    elif not random_init and missing:
        raise RuntimeError(f"could not load blocks {first}-{last - 1} of {repo}: missing {sorted(missing)[:5]}")
    for j, block in enumerate(blocks):
        for mod in block.modules():
            if isinstance(getattr(mod, "layer_idx", None), int):
                mod.layer_idx = first + j
    return blocks


def _compute_in(blocks, dtype):
    """A 4-bit checkpoint names the 16-bit type it was made to compute in, mostly bfloat16, and its 16-bit weights load in that
    type. GPUs without bfloat16 in hardware (a T4) emulate it, several times slower, so the blocks compute in the device's own."""
    import torch
    for block in blocks:
        for mod in block.modules():
            if type(mod).__name__ == "Linear4bit":
                mod.compute_dtype, mod.compute_type_is_set = dtype, True
        for value in [*block.parameters(), *block.buffers()]:
            if value.dtype in (torch.float16, torch.bfloat16) and value.dtype != dtype:
                value.data = value.data.to(dtype)


def _as_in_full_model(loaded: list, config, lay: dict, first: int, device, missing: set, repo: str, saved: str | None = None,
                      dtype=None) -> list:
    """The blocks as the full model builds them, holding the loaded weights. A model built with fewer blocks numbers them from 0
    and sees a smaller total, and some blocks depend on both (T5's position bias lives in block 0; ModernBERT, DiffLlama and
    MiniCPM3 compute constants, layer kinds or rotary tables from the block index or the block count). So the full model is built
    with empty weights (no memory) but real buffers, and each block takes the loaded weights: the tensors move over, not copies."""
    import torch
    from accelerate import init_empty_weights
    with init_empty_weights(include_buffers=False):
        full = auto_class(lay["auto"]).from_config(copy.deepcopy(config))
    stack = full.get_submodule(lay["stack"])
    prefix = lay["stack"] + "."
    out = []
    for j, source in enumerate(loaded):
        i = first + j
        target = stack[i]
        have = dict(source.named_parameters())
        absent = {k[len(f"{prefix}{j}."):] for k in missing if k.startswith(f"{prefix}{j}.")}
        wanted = dict(target.named_parameters())
        lost = [n for n in wanted if n not in have or n in absent or have[n].shape != wanted[n].shape]
        if lost and saved:
            from safetensors import safe_open
            with safe_open(saved, framework="pt") as f:
                names = set(f.keys())
                for n in list(lost):
                    key = f"{prefix}{j}.{n}"
                    if key in names and tuple(f.get_slice(key).get_shape()) == tuple(wanted[n].shape):
                        have[n] = f.get_tensor(key).to(device, dtype or wanted[n].dtype)
                        lost.remove(n)
        if lost:
            raise RuntimeError(f"could not load block {i} of {repo}: no weights for {lost[:5]}")
        stored = set(source.state_dict().keys())
        theirs = dict(source.named_buffers())
        for name in wanted:
            owner, _, leaf = name.rpartition(".")
            (target.get_submodule(owner) if owner else target)._parameters[leaf] = torch.nn.Parameter(have[name].detach(), requires_grad=False)
        for name, buf in list(target.named_buffers()):
            owner, _, leaf = name.rpartition(".")
            value = theirs[name] if name in stored and name in theirs else buf.to(device)
            (target.get_submodule(owner) if owner else target)._buffers[leaf] = value
        out.append(target.eval())
        loaded[j] = None
    return out


def _owner_config(model, stack: str):
    """The config of the nearest module above the stack that has one (DistilBERT's `transformer` has none)."""
    parts = stack.split(".")[:-1]
    while parts:
        config = getattr(model.get_submodule(".".join(parts)), "config", None)
        if config is not None:
            return config
        parts.pop()
    return model.config


def load_ends(repo: str, lay: dict, dtype, device="cpu", random_init: bool = False, log=None, overrides: dict | None = None):
    """The model without its blocks: what runs on the machine with the training code. overrides are config settings such as
    num_labels. Weights the checkpoint does not have (a new classification head) start as transformers initialises them, and
    model._layerpool_new names them."""
    import torch
    stored = _no_quant(load_config(repo))
    config = _overridden(stored, overrides)
    new = []
    picked = _representatives(config, lay["sub"], lay["attr"])
    part = with_blocks(config, lay["sub"], lay["attr"], picked)
    if random_init:
        model = auto_class(lay["auto"]).from_config(part).to(dtype)
        shapes = {n: tuple(p.shape) for n, p in model.named_parameters() if not n.startswith(lay["stack"] + ".")}
        made = random_weights(shapes, config)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in made:
                    p.copy_(made[n].to(p.dtype))
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        out = _range_dir(repo, f"ends {picked}")
        if not os.path.exists(os.path.join(out, "model.safetensors")):
            stack_keys = block_keys(fetch.weight_map(repo), lay["stack"], lay["n"])
            wanted = [k for k in fetch.weight_map(repo) if k not in stack_keys and not k.endswith((".absmax", ".quant_map", ".quant_state", ".nested_absmax",
                                                                                                   ".nested_quant_map")) and "quant_state" not in k]
            _write_checkpoint(out, with_blocks(stored, lay["sub"], lay["attr"], picked), fetch.fetch(repo, wanted, log), lambda k: k)
        model, info = _load(lay["auto"], out, dtype, device, False, overrides)
        prefix = lay["stack"] + "."
        names = {n for n, _ in model.named_parameters()}
        new = sorted(k for k in info.get("missing_keys", []) if not k.startswith(prefix) and k in names)
        base = getattr(model, "base_model_prefix", "")
        inside = base + "." if base and hasattr(model, base) else ""
        lost = [k for k in new if k.startswith(inside) and ".pooler." not in f".{k}"]
        if lost:
            raise RuntimeError(f"could not load the parts of {repo} around its blocks: missing {lost[:5]}")
        if new and log:
            log(f"{repo} has no weights for {', '.join(new)}: they start new and are trained")
    holder = _owner_config(model, lay["stack"])
    full = _sub(config, lay["sub"])
    target = holder if getattr(holder, lay["attr"], None) is not None else _sub(model.config, lay["sub"])
    amap = getattr(target, "attribute_map", {})
    for name in {lay["attr"], amap.get(lay["attr"], lay["attr"])}:
        object.__setattr__(target, name, lay["n"])
    for name in STACK_COUNT_ATTRS:
        if name in vars(target) and vars(target)[name] == len(picked) and len(picked) != lay["n"]:
            object.__setattr__(target, name, lay["n"])
    for key, value in full.__dict__.items():
        if isinstance(value, list) and len(value) == lay["n"]:
            object.__setattr__(target, key, list(value))
    signature = _signatures(full, lay["attr"])
    model.__dict__["_layerpool_kinds"] = {i: picked[[signature[p] for p in picked].index(signature[i])] for i in range(lay["n"])}
    model.__dict__["_layerpool_picked"] = picked
    model.__dict__["_layerpool_new"] = new
    for c in {id(x): x for x in (model.config, model.config.get_text_config(), target)}.values():
        if hasattr(c, "use_cache"):
            object.__setattr__(c, "use_cache", False)
    _no_layerdrop(model, log)
    _generate_without_cache(model)
    _lean_loss(model)
    configs = [model.config] + [v for v in vars(model.config).values() if hasattr(v, "to_dict") and hasattr(v, "model_type")]
    if any(getattr(c, "use_weighted_layer_sum", False) for c in configs):
        raise ValueError("use_weighted_layer_sum needs every block's output on this machine, and the blocks run on the devices; "
                         "turn it off (use_weighted_layer_sum=False)")
    if any(getattr(c, "output_hidden_states", False) or getattr(c, "output_attentions", False) for c in configs):
        raise ValueError(NO_INNER_OUTPUTS)
    return model


NO_INNER_OUTPUTS = ("layerpool cannot give the outputs or attention weights of each block (output_hidden_states, "
                    "output_attentions): the blocks run on the devices, which send back only the last block's output")


def _lean_loss(model):
    """The default causal LM loss, computed in pieces (losses.lean_causal_loss): the machine with the training code holds the
    output layer, and a big vocabulary's logits in float32 take gigabytes there."""
    try:
        from transformers.loss.loss_utils import LOSS_MAPPING
        current = model.loss_function
    except Exception:
        return
    if current is LOSS_MAPPING.get("ForCausalLM"):
        from .losses import lean_causal_loss
        try:
            model.loss_function = lean_causal_loss(current)
        except AttributeError:
            pass


def _generate_without_cache(model):
    """The blocks run on the devices and keep no key/value cache, so generate() runs the whole sequence at each new token
    instead of feeding only the last one to a cache that stays empty."""
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    generate = getattr(model, "generate", None)
    if callable(generate):
        def without_cache(*args, **kwargs):
            kwargs["use_cache"] = False
            return generate(*args, **kwargs)
        model.generate = without_cache


def _no_layerdrop(model, log=None):
    """LayerDrop (wav2vec2, HuBERT, ...) skips random blocks while training; the devices run every block of their range on every
    step, so it is turned off, in the configs and in the modules that copied it."""
    dropped = set()
    configs = [model.config] + [v for v in vars(model.config).values() if hasattr(v, "to_dict") and hasattr(v, "model_type")]
    for c in configs:
        for key, value in list(vars(c).items()):
            if "layerdrop" in key and isinstance(value, (int, float)) and value:
                object.__setattr__(c, key, 0.0)
                dropped.add(key)
    for module in model.modules():
        for key, value in list(vars(module).items()):
            if "layerdrop" in key and isinstance(value, (int, float)) and not isinstance(value, bool) and value:
                setattr(module, key, 0.0)
                dropped.add(key)
    if dropped and log:
        log(f"LayerDrop ({', '.join(sorted(dropped))}) is off: every block runs on every step")


def is_linear(module) -> bool:
    """A layer LoRA can train: torch's Linear (not subclasses that return something else, such as some MoE routers) and
    transformers' Conv1D, GPT-2's linear layer with its weight stored the other way round."""
    import torch
    if isinstance(module, torch.nn.Linear):
        return type(module).forward is torch.nn.Linear.forward or type(module).__name__ in ("Linear4bit", "Linear8bitLt")
    return type(module).__name__ == "Conv1D" and hasattr(module, "nf") and hasattr(module, "nx")


def linear_size(module) -> tuple[int, int]:
    """(inputs, outputs) of a layer is_linear accepts."""
    if hasattr(module, "in_features"):
        return module.in_features, module.out_features
    return module.nx, module.nf


def lora_targets(block) -> list[str]:
    """Every linear layer inside a block, by name (PEFT's "all-linear")."""
    return sorted(name for name, mod in block.named_modules() if name and is_linear(mod))


def targeted(name: str, targets: list[str] | None) -> bool:
    """Whether LoRA goes on this linear layer: every one without target_modules, else the ones named like PEFT's target_modules
    (q_proj matches self_attn.q_proj)."""
    return not targets or any(name == t or name.endswith("." + t) for t in targets)


def lora_init(block_index: int, module_index: int, fan_in: int, fan_out: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(block_index * 1000 + module_index)
    return rng.normal(0, 1 / np.sqrt(fan_in), (rank, fan_in)).astype(np.float32), np.zeros((fan_out, rank), np.float32)


def _dtype_name(t) -> str:
    return str(t.dtype).replace("torch.", "")


def pack_args(per_block: dict, grads: bool = True, carried: dict | None = None) -> tuple[dict, dict]:
    """The arguments the model passed to each block, as JSON plus arrays. Tensors shared by several blocks are stored once.
    Masks are sent as bits. Tensors that need gradients (encoder states, image features) are marked so devices return their
    gradients."""
    import torch
    arrays, seen, sides, infos = {}, {}, [], {}

    def tensor(t):
        if id(t) in seen:
            return {"__t__": seen[id(t)]}
        k = len(seen)
        seen[id(t)] = k
        entry = {"__t__": k}
        name = f"t{k}"
        x = t.detach()
        info = {"dtype": _dtype_name(x), "shape": list(x.shape), "grad": bool(grads and t.requires_grad)}
        if info["grad"]:
            sides.append(t)
            info["side"] = len(sides) - 1
        if x.dtype == torch.bool:
            arrays[name] = np.packbits(x.cpu().numpy().reshape(-1))
            info["bits"] = True
        elif x.is_floating_point() and x.numel() > 1024 and not info["grad"] and bool(((x == 0) | (x <= torch.finfo(x.dtype).min / 2)).all()):
            arrays[name] = np.packbits((x == 0).cpu().numpy().reshape(-1))
            info["bits"], info["additive"] = True, True
        else:
            arrays[name] = (x.float() if x.dtype == torch.bfloat16 else x).cpu().numpy()
        infos[str(k)] = info
        return entry

    carried = carried or {}
    uses: dict[tuple, int] = {}

    def walk(v, block=None):
        if torch.is_tensor(v):
            if id(v) in carried:
                source = carried[id(v)]
                uses[source] = max(uses.get(source, -1), block)
                return {"__carry__": list(source)}
            return tensor(v)
        if isinstance(v, tuple):
            return {"__tuple__": [walk(x, block) for x in v]}
        if isinstance(v, list):
            return {"__list__": [walk(x, block) for x in v]}
        if isinstance(v, dict):
            return {"__dict__": {str(k): walk(x, block) for k, x in v.items()}}
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if type(v).__name__.endswith("Cache"):
            return None
        raise TypeError(f"layerpool cannot send a {type(v).__name__} to the devices (a block argument)")
    meta = {str(i): [walk(list(a), i), walk(dict(kw), i)] for i, (a, kw) in per_block.items()}
    out = {"blocks": meta, "tensors": infos, "sides": len(sides)}
    if uses:
        out["carry"] = [[i, k, last] for (i, k), last in sorted(uses.items())]
    return out, arrays, sides


def unpack_args(meta: dict, arrays: dict, device, dtype, wanted=None) -> tuple[dict, dict]:
    """Back to tensors on this device. Returns {block: (args, kwargs)} and {side index: leaf tensor needing a gradient}."""
    import torch
    made, leaves = {}, {}

    def tensor(entry):
        k = entry["__t__"]
        if k in made:
            return made[k]
        info = meta["tensors"][str(k)]
        raw = arrays[f"t{k}"]
        shape = tuple(info["shape"])
        target = getattr(torch, info["dtype"])
        if info.get("bits"):
            flat = np.unpackbits(raw)[: int(np.prod(shape))].astype(bool).reshape(shape)
            if info.get("additive"):
                out_dtype = dtype if target.is_floating_point else target
                t = torch.zeros(shape, dtype=out_dtype).masked_fill(~torch.from_numpy(flat), torch.finfo(out_dtype).min)
            else:
                t = torch.from_numpy(flat)
        else:
            t = torch.from_numpy(np.asarray(raw)).to(target)
            if t.is_floating_point():
                t = t.to(dtype)
        t = t.to(device)
        if info.get("grad"):
            t = t.requires_grad_()
            leaves[info["side"]] = t
        made[k] = t
        return t

    def walk(v):
        if isinstance(v, dict):
            if "__t__" in v:
                return tensor(v)
            if "__tuple__" in v:
                return tuple(walk(x) for x in v["__tuple__"])
            if "__list__" in v:
                return [walk(x) for x in v["__list__"]]
            if "__dict__" in v:
                return {k: walk(x) for k, x in v["__dict__"].items()}
        return v
    out = {}
    for key, (a, kw) in meta["blocks"].items():
        if wanted is not None and int(key) not in wanted:
            continue
        out[int(key)] = (tuple(walk(a)), walk(kw))
    return out, leaves


def mlx_masks(meta: dict | None, arrays: dict, wanted) -> dict | None:
    """Where each token may look in each wanted block, {block: (rows, 1, tokens, tokens) bool array}, or which tokens are real
    ((rows, tokens), what linear-attention blocks get), from the masks the model gave its blocks; None when it gave none, so the
    blocks are plainly causal. MLX blocks take this instead of the torch arguments."""
    if not meta or not meta.get("blocks"):
        return None
    import torch
    blocks = {}
    for key, (_, kw) in meta["blocks"].items():
        inner = kw.get("__dict__", kw) if isinstance(kw, dict) else {}
        if isinstance(inner, dict):
            kept = {k: inner[k] for k in ("attention_mask", "position_ids") if isinstance(inner.get(k), dict)}
            if kept:
                blocks[key] = ([], {"__dict__": kept})
    if not blocks:
        return None
    got, _ = unpack_args({**meta, "blocks": blocks}, arrays, "cpu", torch.float32, wanted)
    out, made = {}, {}
    for i, (_, kw) in got.items():
        positions = kw.get("position_ids")
        if torch.is_tensor(positions) and positions.dim() >= 1:
            plain = torch.arange(positions.shape[-1]).expand_as(positions)
            if not torch.equal(positions.long(), plain):
                raise ValueError("MLX blocks number the tokens of each row 0, 1, 2, ... themselves, and this batch gives other "
                                 "positions (several sequences packed into one row?); train it without packing, or start the Mac "
                                 "with --backend torch")
        m = kw.get("attention_mask")
        if not torch.is_tensor(m):
            continue
        if id(m) not in made:
            if m.dim() == 2:
                attend = m if m.dtype == torch.bool else m != 0
            else:
                attend = m if m.dtype == torch.bool else (m > m.min() / 2 if m.min() < 0 else torch.ones_like(m, dtype=torch.bool))
            made[id(m)] = attend.detach().numpy()
        out[i] = made[id(m)]
    return out or None


def _with_carried(v, carry: dict):
    """Block arguments with each reference to an earlier block's output replaced by that output."""
    if isinstance(v, dict):
        if "__carry__" in v:
            return carry[tuple(v["__carry__"])]
        return {k: _with_carried(x, carry) for k, x in v.items()}
    if isinstance(v, tuple):
        return tuple(_with_carried(x, carry) for x in v)
    if isinstance(v, list):
        return [_with_carried(x, carry) for x in v]
    return v


def first_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _attach_lora(base, A, B, scale: float, core):
    """LoRA on a linear layer without replacing it: a forward hook adds scale * B(A x) to its output, so the layer keeps its
    class and attributes for the model code around it."""
    def hook(module, inputs, output):
        if not core.adapter or not hasattr(output, "dtype"):
            return output
        x = inputs[0]
        return output + (scale * ((x.float() @ A.T) @ B.T)).to(output.dtype)
    handle = base.register_forward_hook(hook)
    handles = base.__dict__.setdefault("_layerpool_lora", [])
    handles.append(handle)


def unwrap_lora(blocks):
    """The blocks without the LoRA of an earlier run, to put new ones on them."""
    for block in blocks:
        for mod in block.modules():
            for handle in mod.__dict__.pop("_layerpool_lora", []):
                handle.remove()
    return blocks


class BlockCore:
    """A slice of the stack on one torch device: forward, backward, its own AdamW, and its state for checkpoints."""

    def __init__(self, blocks, first: int, rank: int, scale: float, lr: float, loss_scale: float = 1.0, checkpointing: bool = True,
                 mode: str = "lora", weight_decay: float = 0.0, targets: list[str] | None = None):
        import torch
        self.layers, self.first = blocks, first
        self.adapter = True
        self.loss_scale, self.checkpointing, self.mode = loss_scale, checkpointing, mode

        self.device = next(blocks[0].parameters()).device
        floats = [p for p in blocks[0].parameters() if p.is_floating_point()]
        self.dtype = floats[0].dtype if floats else torch.float32
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            self.dtype = torch.float16
        if self.dtype == torch.float32:
            self.loss_scale = 1.0
        self.max_loss_scale, self.good_steps, self.overflows = max(1.0, self.loss_scale), 0, 0
        self.params, self.init = {}, {}
        if mode == "full":
            for j, block in enumerate(blocks):
                for n, p in block.named_parameters():
                    self.params[f"{first + j}.{n}"] = p
                    p.requires_grad_(True)
        else:
            for j, block in enumerate(blocks):
                i = first + j
                for k, name in enumerate(lora_targets(block)):
                    if not targeted(name, targets):
                        continue
                    base = block.get_submodule(name)
                    fan_in, fan_out = linear_size(base)
                    A, B = lora_init(i, k, fan_in, fan_out, rank)
                    d = next(base.parameters()).device
                    A, B = torch.nn.Parameter(torch.tensor(A, device=d)), torch.nn.Parameter(torch.tensor(B, device=d))
                    _attach_lora(base, A, B, scale, self)
                    self.params[f"{i}.{name}.A"], self.params[f"{i}.{name}.B"] = A, B
                    self.init[f"{i}.{name}"] = (i, k, fan_in, fan_out, rank)
        if not self.params and targets:
            raise ValueError(f"target_modules {targets} names none of the linear layers in {type(blocks[0]).__name__}: "
                             f"{', '.join(lora_targets(blocks[0]))}")
        if not self.params:
            raise ValueError(f"layerpool found no linear layers to put LoRA on in {type(blocks[0]).__name__}; train this model with "
                             "full=True (mode=\"full\") instead")
        decayed = [p for p in self.params.values() if p.dim() > 1]
        plain = [p for p in self.params.values() if p.dim() <= 1]
        groups = [{"params": decayed, "weight_decay": weight_decay}] + ([{"params": plain, "weight_decay": 0.0}] if plain else [])
        self.opt = torch.optim.AdamW(groups, lr=lr, betas=BETAS, eps=EPS)
        self.graphs = {}
        self.initial_state = {k: v.detach().float().cpu().numpy().copy() for k, v in self.params.items()} if mode == "full" else None

    @property
    def last(self) -> int:
        return self.first + len(self.layers)

    def tensor(self, array: np.ndarray):
        import torch
        return torch.from_numpy(np.ascontiguousarray(array))

    def numpy(self, value) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    def set_adapter(self, enabled: bool):
        self.adapter = enabled

    def _run(self, h, args: dict, carry: dict | None = None, plan: list | None = None):
        """The slice on h. carry holds outputs of earlier blocks that later blocks take as arguments (T5's position bias), by
        (block, place in its output); plan lists [block, place, last block using it] for the whole model."""
        import torch
        import torch.utils.checkpoint as ckpt
        carry = dict(carry or {})
        for j, block in enumerate(self.layers):
            index = self.first + j
            a, kw = args[index]
            if plan:
                a, kw = _with_carried(a, carry), _with_carried(kw, carry)
            want = [k for i, k, _ in plan or [] if i == index]
            d = next(block.parameters()).device
            h = h.to(d)

            def call(x, block=block, a=a, kw=kw, want=want):
                out = block(x, *a, **kw)
                return (first_tensor(out), *[out[k] for k in want]) if want else first_tensor(out)
            res = ckpt.checkpoint(call, h, use_reentrant=False) if self.checkpointing and torch.is_grad_enabled() else call(h)
            if want:
                h = res[0]
                for k, value in zip(want, res[1:]):
                    carry[(index, k)] = value
            else:
                h = res
        if plan is None:
            return h
        onward = {(i, k) for i, k, last in plan if last >= self.last}
        return h, {key: v for key, v in carry.items() if key in onward}

    def _here(self, value):
        """A carried tensor on this slice's device, in its precision when it holds numbers (the whole model, run in one
        precision, would have made it in that one)."""
        return value.to(self.device, self.dtype) if value.is_floating_point() else value.to(self.device)

    def forward(self, x, args: dict, carry: dict | None = None, plan: list | None = None, key=None, leaves: dict | None = None):
        """The slice's output; with a plan (a model whose blocks hand tensors on), also the carried tensors later slices need.
        With a key (a micro-batch whose backward comes later), the forward pass keeps its graph for backward(key=...): with
        checkpointing only each block's input, so the backward runs every block forward once more, not twice."""
        import torch
        if key is None:
            with torch.no_grad():
                if plan is None:
                    return self._run(x.to(self.device, self.dtype), args)
                carry = {k: self._here(v) for k, v in (carry or {}).items()}
                return self._run(x.to(self.device, self.dtype), args, carry, plan)
        received = None
        with torch.enable_grad():
            inp = x.to(self.device, self.dtype).detach().requires_grad_(True)
            if plan is None:
                out, onward = self._run(inp, args), None
            else:
                received = {k: self._here(v).detach().requires_grad_(v.is_floating_point()) for k, v in (carry or {}).items()}
                out, onward = self._run(inp, args, received, plan)
        self.graphs[key] = (inp, out, onward, received, leaves or {}, args)
        if plan is None:
            return out.detach()
        return out.detach(), {k: v.detach() for k, v in onward.items()}

    def backward(self, x, args: dict, leaves: dict, grad_out, carry: dict | None = None, carry_grads: dict | None = None,
                 plan: list | None = None, key=None):
        """Gradient of the stage input, and of every argument tensor that needs one, for the given output gradient. With a plan,
        also takes the gradients of the carried tensors this slice passed on, and returns those of the ones it received.
        The trained weights' gradients add up unscaled in float32. When a 16-bit backward overflows, the loss scale halves and
        this micro-batch's backward runs again, so the step goes on instead of being skipped on every device."""
        import torch
        kept = self.graphs.pop(key, None) if key is not None else None
        if kept is None and args is None:
            raise RuntimeError("layerpool: a backward came for a micro-batch whose forward graph is gone")
        if kept is not None:
            inp, out, onward, received, leaves, args = kept
        else:
            inp = x.to(self.device, self.dtype).detach().requires_grad_(True)
            received = {k: self._here(v).detach().requires_grad_(v.is_floating_point()) for k, v in (carry or {}).items()} \
                if plan is not None else None
            out = onward = None
        trained = list(self.params.values())
        inputs = [t for t in [inp, *leaves.values(), *(received or {}).values()] if t.requires_grad]
        while True:
            with torch.enable_grad():
                if out is None:
                    out, onward = (self._run(inp, args), None) if plan is None else self._run(inp, args, received, plan)
                outs, grads = [out], [self._scaled(grad_out, out)]
                for k, g in (carry_grads or {}).items() if plan is not None else ():
                    if k in onward and onward[k].requires_grad:
                        outs.append(onward[k])
                        grads.append(self._scaled(g, onward[k]))
                if self.loss_scale == 1.0:
                    for t in inputs:
                        t.grad = None
                    torch.autograd.backward(outs, grads)
                    found = {id(t): t.grad for t in inputs if t.grad is not None}
                    break
                targets = inputs + [p for p in trained if p.requires_grad]
                got = torch.autograd.grad(outs, targets, grads, allow_unused=True)
            found = {id(t): g for t, g in zip(targets, got) if g is not None}
            if self.loss_scale <= 1.0 or bool(torch.stack([torch.isfinite(g).all().to(self.device) for g in found.values()]).all()):
                break
            self.loss_scale, self.good_steps, out = self.loss_scale / 2, 0, None
            self.overflows += 1
        for p in trained:
            if id(p) in found:
                g = self._unscaled(found[id(p)])
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.add_(g)
        back = lambda t: self._unscaled(found[id(t)]) if id(t) in found else None
        sides = {k: back(v) for k, v in leaves.items()}
        if plan is None:
            return back(inp), sides
        return back(inp), sides, {k: back(v) for k, v in received.items() if id(v) in found}

    def _scaled(self, grad, like):
        """The loss scale goes on in float32, before the cast to a 16-bit dtype, so small gradients survive it."""
        return (grad.to(like.device).float() * self.loss_scale).to(like.dtype)

    def _unscaled(self, grad):
        return grad.detach().float() / self.loss_scale

    def grad_sq(self) -> float:
        total = 0.0
        for p in self.params.values():
            if p.grad is not None:
                total += float(p.grad.float().pow(2).sum())
        return total

    def step(self, lr: float | None = None, clip: float = 1.0, skip: bool = False):
        if skip:
            self.opt.zero_grad(set_to_none=True)
            return
        for p in self.params.values():
            if p.grad is not None:
                p.grad *= clip
        if lr is not None:
            for group in self.opt.param_groups:
                group["lr"] = lr
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.good_steps += 1
        if self.good_steps >= SCALE_GROWTH_STEPS and self.loss_scale < self.max_loss_scale:
            self.loss_scale, self.good_steps = min(self.max_loss_scale, self.loss_scale * 2), 0

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)
        self.graphs.clear()

    def set_weight_decay(self, value: float):
        """Weight decay for the matrices; norms and biases (one-dimensional) never decay, as in the Trainer."""
        self.opt.param_groups[0]["weight_decay"] = value

    def state(self) -> dict[str, np.ndarray]:
        out = {}
        for k, p in self.params.items():
            out[k] = p.detach().float().cpu().numpy()
            st = self.opt.state.get(p)
            if st:
                out[k + ".m"] = st["exp_avg"].detach().float().cpu().numpy()
                out[k + ".v"] = st["exp_avg_sq"].detach().float().cpu().numpy()
                out[k + ".t"] = np.array([float(st["step"])], np.float32)
        return out

    def weights(self) -> dict[str, np.ndarray]:
        return {k: p.detach().float().cpu().numpy() for k, p in self.params.items()}

    def initial(self) -> dict[str, np.ndarray]:
        if self.mode == "full":
            return dict(self.initial_state)
        out = {}
        for key, args in self.init.items():
            out[key + ".A"], out[key + ".B"] = lora_init(*args)
        return out

    def load_state(self, saved: dict[str, np.ndarray]) -> int:
        import torch
        restored = 0
        for k, p in self.params.items():
            if k in saved:
                with torch.no_grad():
                    p.data.copy_(torch.tensor(saved[k], device=p.device, dtype=p.dtype))
                restored += 1
                self.opt.state.pop(p, None)
                if k + ".m" in saved:
                    self.opt.state[p] = {"step": torch.tensor(float(saved[k + ".t"][0])),
                                         "exp_avg": torch.tensor(saved[k + ".m"], device=p.device, dtype=p.dtype),
                                         "exp_avg_sq": torch.tensor(saved[k + ".v"], device=p.device, dtype=p.dtype)}
        return restored


"""Before a run, a small copy of the model's own architecture is split the way the run will split it and compared with the whole
copy. A model whose split would compute something else (an architecture layerpool does not handle yet, or one added to
transformers later) is stopped here with a clear message instead of training wrong. Each architecture is checked once per
transformers version."""
import json
import os
import tempfile

from . import __version__
from .blocks import BlockCore, layout, load_blocks, load_config, load_ends
from .codec import replace_file
from .driver import Ends, LocalChain

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "layerpool", "checked.json")
SIZE = {
    "hidden_size": 64, "d_model": 64, "n_embd": 64, "embed_dim": 64, "dim": 64, "n_embed": 64, "out_hidden_size": 64,
    "intermediate_size": 128, "ffn_dim": 128, "encoder_ffn_dim": 128, "decoder_ffn_dim": 128, "d_ff": 128, "n_inner": 128,
    "hidden_dim": 128, "decoder_intermediate_size": 128,
    "num_attention_heads": 4, "n_head": 4, "encoder_attention_heads": 4, "decoder_attention_heads": 4, "num_heads": 4,
    "n_heads": 4, "decoder_num_attention_heads": 4,
    "num_key_value_heads": 2, "num_kv_heads": 2, "multi_query_group_num": 2, "decoder_num_key_value_heads": 2,
    "head_dim": 16, "kv_channels": 16, "d_kv": 16,
    "num_experts": 4, "num_local_experts": 4, "n_routed_experts": 4, "moe_num_experts": 4, "num_experts_per_tok": 2,
    "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64, "n_shared_experts": 1, "n_group": 1, "topk_group": 1,
    "moe_topk": 2, "kv_lora_rank": 16, "q_lora_rank": 32, "qk_rope_head_dim": 8, "qk_nope_head_dim": 8, "v_head_dim": 16,
    "state_size": 8, "ssm_state_size": 8, "d_state": 8, "mamba_d_state": 8,
}
LAYERS = ("num_hidden_layers", "n_layer", "num_layers", "decoder_layers", "num_decoder_layers", "n_layers", "encoder_layers",
          "num_encoder_layers", "depth", "decoder_num_hidden_layers", "text_layers")
BLOCKS = 4


def _small(d: dict, side: bool = False) -> dict:
    """The config's sizes made tiny; its structure (kinds of layers, experts, attention patterns) stays."""
    out = dict(d)
    counts = {d[k] for k in LAYERS if isinstance(d.get(k), int) and not isinstance(d.get(k), bool) and d[k] > 1}
    period = max([v for v in (d.get("sliding_window_pattern"), d.get("full_attention_interval")) if isinstance(v, int)] or [0])
    blocks = period if period > BLOCKS and period in range(BLOCKS + 1, 13) else BLOCKS
    for k, v in d.items():
        if isinstance(v, dict) and k.endswith("_config") and "model_type" in v:
            out[k] = _small(v, side or any(w in k for w in ("vision", "audio", "speech", "encoder", "perceiver", "qformer", "image")))
        elif isinstance(v, dict) and k in ("rope_parameters", "rope_scaling") and "mrope_section" in v:
            out[k] = {**v, "mrope_section": [2, 3, 3]}
        elif k in SIZE and isinstance(v, int) and not isinstance(v, bool):
            out[k] = 64 if side and k in ("intermediate_size", "ffn_dim", "d_ff", "hidden_dim", "encoder_ffn_dim") else SIZE[k]
        elif k in LAYERS and isinstance(v, int) and not isinstance(v, bool) and v > 1:
            out[k] = 2 if side or k in ("encoder_layers", "num_encoder_layers") else min(v, blocks)
        elif isinstance(v, list) and len(v) in counts and len(v) > blocks and blocks > BLOCKS:
            out[k] = list(v[:blocks])
        elif isinstance(v, list) and len(v) in counts and len(v) > BLOCKS:
            kinds = []
            for i, x in enumerate(v):
                if x not in [v[j] for j in kinds]:
                    kinds.append(i)
            kinds = kinds[:BLOCKS]
            picked = sorted(kinds + [i for i in range(len(v)) if i not in kinds][:BLOCKS - len(kinds)])
            out[k] = [v[i] for i in picked]
    for k in ("quantization_config", "_name_or_path"):
        out.pop(k, None)
    return out


def _inputs(model, config, auto: str):
    """Inputs the model takes, by its kind: token ids, pixels, audio features or a waveform."""
    import torch
    g = torch.Generator().manual_seed(2)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    vocab = min(getattr(text, "vocab_size", None) or 300, 300)
    ids = torch.randint(3, vocab, (2, 12), generator=g)
    extra = {"decoder_input_ids": torch.randint(3, vocab, (2, 6), generator=g)} if getattr(config, "is_encoder_decoder", False) else {}
    main = getattr(model, "main_input_name", "input_ids")
    if main == "input_ids" or auto == "AutoModelForImageTextToText":
        eos = getattr(text, "eos_token_id", None)
        eos = eos[0] if isinstance(eos, list) and eos else eos
        if isinstance(eos, int) and 0 <= eos < (getattr(text, "vocab_size", None) or 300):
            ids[:, -1] = eos
        yield {"input_ids": ids, "use_cache": False, **extra}
        yield {"input_ids": ids, **extra}
    elif main == "pixel_values":
        vision = getattr(config, "vision_config", None) or config
        size = getattr(vision, "image_size", None) or 224
        h, w = (size[0], size[-1]) if isinstance(size, (list, tuple)) else (size.get("height", 224), size.get("width", 224)) \
            if isinstance(size, dict) else (size, size)
        yield {"pixel_values": torch.randn(2, getattr(vision, "num_channels", None) or 3, h, w, generator=g), **extra}
    elif main == "input_features":
        mel = getattr(config, "num_mel_bins", None) or 80
        positions = getattr(config, "max_source_positions", None)
        if positions:
            yield {"input_features": torch.randn(2, mel, 2 * positions, generator=g), **extra}
        yield {"input_features": torch.randn(2, 64, mel, generator=g), **extra}
    elif main == "input_values":
        yield {"input_values": torch.randn(2, 4000, generator=g), **extra}


def _output(out):
    for key in ("logits", "last_hidden_state", "prediction_logits", "pooler_output", "image_embeds", "text_embeds"):
        value = out.get(key) if isinstance(out, dict) else getattr(out, key, None)
        if value is not None and hasattr(value, "shape"):
            return value
    return out[0] if isinstance(out, (tuple, list)) else out


SIZE_KEYS = set(SIZE) | set(LAYERS) | {"vocab_size", "max_position_embeddings", "layer_types", "_name_or_path", "transformers_version",
                                       "torch_dtype", "dtype", "quantization_config", "architectures"}


def settings_hash(config) -> str:
    """What decides how an architecture computes, beyond its kind: rope settings, activation, attention pattern, flags. Sizes,
    names and the vocabulary are left out, since the small copy replaces them. Results are remembered per kind and this hash, so
    two models of one kind with different settings (Phi-3 and Phi-3.5's longrope) are each checked."""
    import hashlib

    def keep(d):
        out = {}
        for k, v in sorted(d.items()):
            if k in SIZE_KEYS:
                continue
            out[k] = keep(v) if isinstance(v, dict) else v
        return out
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return hashlib.sha1(json.dumps(keep(data), sort_keys=True, default=str).encode()).hexdigest()[:12]


def kept_rng(fn):
    """Checks draw random numbers; the training code's own random state (its seed) is left as it was."""
    def wrapped(*a, **kw):
        import random

        import numpy as np
        import torch
        states = (torch.get_rng_state(), random.getstate(), np.random.get_state())
        try:
            return fn(*a, **kw)
        finally:
            torch.set_rng_state(states[0])
            random.setstate(states[1])
            np.random.set_state(states[2])
    return wrapped


def _remembered(key: str):
    try:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f).get(key)
    except (OSError, ValueError):
        return None


def _remember(key: str, result: str):
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        try:
            with open(CACHE, encoding="utf-8") as f:
                known = json.load(f)
        except (OSError, ValueError):
            known = {}
        known[key] = result
        tmp = f"{CACHE}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(known, f, indent=1)
        replace_file(tmp, CACHE)
    except OSError:
        pass


@kept_rng
def check(repo: str, auto: str, log=print) -> set:
    """Raises ValueError when this architecture does not split correctly; otherwise returns what else was learnt about it:
    "whole_batches" (its blocks give other results when a batch is cut into pieces) and "place" (its blocks depend on where
    they sit in the model, which the 4-bit path does not rebuild). Returns nothing when no small copy can be made here (then it
    says so)."""
    from transformers.utils import logging
    level, bars = logging.get_verbosity(), logging.is_progress_bar_enabled()
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    try:
        return _check(repo, auto, log) or set()
    finally:
        logging.set_verbosity(level)
        if bars:
            logging.enable_progress_bar()


def _check(repo: str, auto: str, log) -> set | None:
    import torch
    import transformers
    config = load_config(repo)
    kind = str(getattr(config, "model_type", ""))
    key = f"{kind}|{settings_hash(config)}|{auto}|transformers {transformers.__version__}|layerpool {__version__}"
    known = _remembered(key)
    if known and known.startswith("ok"):
        return set(known.split(":")[1:])
    if known and known.startswith("wrong:"):
        raise ValueError(known[6:])
    with tempfile.TemporaryDirectory(prefix="layerpool_check_") as folder:
        try:
            small = type(config).from_dict(_small(config.to_dict()))
            torch.manual_seed(0)
            whole = getattr(transformers, auto).from_config(small).float().eval()
            with torch.no_grad():
                for name, p in whole.named_parameters():
                    if "norm" in name and p.is_floating_point():
                        p.add_(torch.randn_like(p) * 0.3)
            whole.save_pretrained(folder)
            want, inputs = None, None
            for candidate in _inputs(whole, small, auto):
                try:
                    with torch.no_grad():
                        want = _output(whole(**candidate))
                        again = _output(whole(**candidate))
                    inputs = candidate
                    break
                except Exception:
                    continue
            if inputs is None:
                raise RuntimeError("no inputs fit the small copy")
            if not torch.equal(want, again):
                log(f"note: {kind} gives different outputs for the same input (it draws random numbers), so its split cannot be "
                    "checked by comparison; training goes on")
                return
            lay = layout(load_config(folder), auto)
            n = lay["n"]
            cuts = sorted({0, 1, max(1, n - 1), n})
            cores = [BlockCore(load_blocks(folder, lay, a, b, "cpu", torch.float32), a, 4, 2.0, 1e-3, mode="full")
                     for a, b in zip(cuts, cuts[1:])]
            model = load_ends(folder, lay, torch.float32)
            Ends(model, lay, LocalChain(cores))
            model.eval()
        except Exception as e:
            log(f"note: layerpool could not make a small copy of {kind} to check its split ({type(e).__name__}); training goes on")
            return
        try:
            with torch.no_grad():
                got = _output(model(**inputs))
        except Exception as e:
            text = f"layerpool cannot split {kind} models yet: split across devices, the model fails ({type(e).__name__}: {str(e)[:200]})"
            _remember(key, "wrong:" + text)
            raise ValueError(text) from None
        gap = float((got.float() - want.float()).abs().max())
        scale = float(want.float().abs().max()) or 1.0
        if gap > 1e-4 * max(1.0, scale):
            text = (f"layerpool cannot split {kind} models correctly yet: on a small copy of this architecture the split model's "
                    f"outputs differ from the whole model's (by {gap:.3g}), so training would not be right")
            _remember(key, "wrong:" + text)
            raise ValueError(text)
        notes = []
        try:
            if _cuts_change_results(whole, folder, auto, lay):
                notes.append("whole_batches")
                log(f"note: {kind} blocks compute differently when a batch is cut into pieces, so its batches go to the devices whole")
            if _depends_on_place(whole, folder, auto, lay):
                notes.append("place")
        except Exception:
            pass
        _remember(key, "ok" + "".join(f":{n}" for n in notes))
        return set(notes)


def _block_calls(whole, lay: dict, inputs: dict) -> dict:
    """What each block is given when the whole model runs on these inputs: {block: (input, args, kwargs)}."""
    import torch
    seen = {}
    blocks = whole.get_submodule(lay["stack"])

    def grab(i):
        def hook(module, args, kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in ("past_key_values", "past_key_value", "cache_params")}
            if args:
                seen[i] = (args[0].detach(), tuple(args[1:]), kwargs)
            else:
                kwargs = dict(kwargs)
                seen[i] = (kwargs.pop("hidden_states").detach(), (), kwargs)
        return hook
    hooks = [b.register_forward_pre_hook(grab(i), with_kwargs=True) for i, b in enumerate(blocks)]
    try:
        with torch.no_grad():
            whole(**inputs)
    finally:
        for h in hooks:
            h.remove()
    return seen


def _text_inputs(whole, rows: int) -> dict | None:
    import torch
    if getattr(whole, "main_input_name", "input_ids") != "input_ids" or getattr(whole.config, "is_encoder_decoder", False):
        return None
    vocab = min(getattr(whole.config.get_text_config(), "vocab_size", 300) or 300, 300)
    return {"input_ids": torch.randint(3, vocab, (rows, 12), generator=torch.Generator().manual_seed(5)), "use_cache": False}


def _cuts_change_results(whole, folder: str, auto: str, lay: dict) -> bool:
    """Whether the blocks give other results when a batch is cut into micro-batches the way the devices get them. The batch has
    as many rows as the small copy has heads, so a tensor whose size merely equals the heads is cut too if the cutting rule
    mistakes it for the batch (MPT's and Bloom's attention biases)."""
    import torch
    from .driver import split_args
    rows = 4
    inputs = _text_inputs(whole, rows)
    if inputs is None:
        return False
    calls = _block_calls(whole, lay, inputs)
    blocks = whole.get_submodule(lay["stack"])
    first = min(calls)
    per_block = {i: (a, kw) for i, (_, a, kw) in calls.items()}

    def through(h, trees):
        for i in sorted(trees):
            a, kw = trees[i]
            h = blocks[i](h, *a, **kw)
            h = h[0] if isinstance(h, tuple) else h
        return h
    with torch.no_grad():
        try:
            want = through(calls[first][0], per_block)
        except Exception:
            return False
        cuts = [(0, 2), (2, rows)]
        trees, _ = split_args(per_block, rows, cuts)
        try:
            got = torch.cat([through(calls[first][0][a:b], tree) for (a, b), tree in zip(cuts, trees)])
        except Exception:
            return True
    return not torch.allclose(got, want, atol=1e-4, rtol=1e-4)


def _depends_on_place(whole, folder: str, auto: str, lay: dict) -> bool:
    """Whether a block computes differently when transformers builds it in a model of just its range (numbered from 0, with a
    smaller total) than in the whole model. The 16-bit path rebuilds blocks inside the whole model; the 4-bit path cannot."""
    import torch
    inputs = _text_inputs(whole, 2)
    if inputs is None or lay["n"] < 2:
        return False
    calls = _block_calls(whole, lay, inputs)
    i = lay["n"] - 1
    if i not in calls:
        return False
    alone = load_blocks(folder, lay, i, i + 1, "cpu", torch.float32, in_place=False)[0]
    h, a, kw = calls[i]
    with torch.no_grad():
        want = whole.get_submodule(lay["stack"])[i](h, *a, **kw)
        try:
            got = alone(h, *a, **kw)
        except Exception:
            return True
    want = want[0] if isinstance(want, tuple) else want
    got = got[0] if isinstance(got, tuple) else got
    return not torch.allclose(got, want, atol=1e-5, rtol=1e-4)

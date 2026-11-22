import glob
import json
import os
import struct


from .blocks import block_keys, layout, linear_size, load_config as hf_config, lora_targets, skeleton, targeted



def tensor_bytes(repo: str) -> dict[str, int]:
    out = {}
    if os.path.isdir(repo):
        for path in glob.glob(os.path.join(repo, "*.safetensors")):
            with open(path, "rb") as f:
                header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            out.update({k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in header.items() if k != "__metadata__"})
        return out
    from huggingface_hub import get_safetensors_metadata
    for meta in get_safetensors_metadata(repo).files_metadata.values():
        out.update({k: v.data_offsets[1] - v.data_offsets[0] for k, v in meta.tensors.items()})
    return out


def load_config(repo: str) -> dict:
    if os.path.isdir(repo):
        path = os.path.join(repo, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _block_shape(config, lay: dict) -> tuple[int, int]:
    """(weights, LoRA in+out sum per unit of rank) of one block, from the model skeleton."""
    block = skeleton(config, lay["auto"]).get_submodule(lay["stack"])[0]
    params = sum(p.numel() for p in block.parameters())
    lora = 0
    for name in lora_targets(block):
        mod = block.get_submodule(name)
        lora += sum(linear_size(mod))
    return params, lora


def estimate(repo: str, lay: dict, rank: int, seq: int, batch: int, micro: int, mode: str = "lora", config=None, stack: str | None = None) -> dict:
    """Memory of one block on a device: its weights (as stored), the trainable part with its AdamW state, and the saved inputs of
    every micro-batch in flight."""
    config = config or hf_config(repo)
    text = config.get_text_config()
    hidden = getattr(text, "hidden_size", None) or getattr(config, "d_model", 1024)
    inner = getattr(text, "intermediate_size", None) or 4 * hidden
    params, lora = _block_shape(config, lay)
    saved_inputs = micro * batch * seq * hidden * 2
    if mode == "full":
        per_block = params * 16
    else:
        try:
            sizes = tensor_bytes(repo)
            index = block_keys(sizes, stack or lay["stack"], lay["n"])
            per = [0] * lay["n"]
            for key, i in index.items():
                per[i] += sizes[key]
            stored = max(per)
        except Exception:
            stored = params * 2
        per_block = stored + lora * rank * 4 * 4
    peak = batch * seq * (3 * inner + 6 * hidden) * 2 + batch * seq * seq * 2
    return {"layers": lay["n"], "layer_gb": round((per_block + saved_inputs) / 1e9, 4), "overhead_gb": round(0.8 + peak / 1e9, 3)}




QUANTIZERS = ("unsloth",)


def guess_mlx_repo(repo: str) -> str | None:
    """The mlx-community 4-bit copy of a bitsandbytes repo that only quantizes a published model (unsloth's). Any other repo
    may be a fine-tune with the base model's name, so its MLX weights must be given as mlx_model=."""
    org, _, name = repo.rpartition("/")
    if org.lower() not in QUANTIZERS:
        return None
    for suffix in ("-unsloth-bnb-4bit", "-bnb-4bit", "-4bit", "-bnb"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    candidate = f"mlx-community/{name}-4bit"
    try:
        from huggingface_hub import model_info
        model_info(candidate)
        return candidate
    except Exception:
        return None


RUN_EXTRAS = ("name", "checkpointing", "peer_replicas", "rebalance", "rebalance_cost_s", "streams", "target_modules")


def build_job(model: str, mlx_model: str | None = None, mode: str = "lora", rank: int = 16, alpha: float | None = None, lr: float = 1e-4,
              schedule: str = "cosine", warmup: int = 0, min_lr_ratio: float = 0.1, weight_decay: float = 0.0, micro: int = 8,
              micro_batch_size: int | None = None, codec: str = "fp16", ckpt_every: int = 4, random_init: bool = False, expected_steps: int | None = None,
              seq: int = 512, auto: str = "AutoModelForCausalLM", **extra) -> dict:
    if mode not in ("lora", "full"):
        raise ValueError("mode must be 'lora' or 'full'")
    raw = load_config(model)
    if "quantization" in raw:
        raise ValueError(f"{model} is an MLX-only model; pass the Hugging Face model as model= and this one as mlx_model=")
    quant = raw.get("quantization_config") or {}
    if quant and not (quant.get("quant_method") == "bitsandbytes" and quant.get("load_in_4bit")):
        raise ValueError(f"{model} is quantized with {quant.get('quant_method', 'an unknown method')}; layerpool trains the 16-bit model "
                         "or a bitsandbytes 4-bit one (QLoRA)")
    kind = "q4" if quant else "plain"
    if kind == "q4" and mode == "full":
        raise ValueError(f"{model} is a 4-bit model; full fine-tuning needs the 16-bit version (LoRA works on 4-bit)")
    if random_init and mode != "full":
        raise ValueError("training from scratch (random_init=True) needs mode='full'")
    config = hf_config(model)
    lay = layout(config, auto)
    text = config.get_text_config()
    model_type = str(getattr(text, "model_type", raw.get("model_type", "")))
    causal = auto == "AutoModelForCausalLM"
    mlx_ok = causal
    guessed = None
    if kind == "q4" and mlx_model is None and mlx_ok:
        mlx_model = guessed = guess_mlx_repo(model)
    pad = getattr(text, "pad_token_id", None)
    if pad is None:
        eos = getattr(text, "eos_token_id", None)
        pad = eos[0] if isinstance(eos, list) else (eos or 0)
    vocab = getattr(text, "vocab_size", None)
    if vocab and not 0 <= int(pad) < vocab:
        pad = 0
    sizes = {"torch": estimate(model, lay, rank, seq, micro_batch_size or 1, micro, mode, config)}
    if mlx_model and mlx_ok:
        try:
            sizes["mlx"] = estimate(mlx_model, lay, rank, seq, micro_batch_size or 1, micro, mode, config)
        except Exception:
            pass
    job = {"repo": model, "mlx_repo": mlx_model if mlx_ok else None, "kind": kind, "mode": mode, "random_init": random_init, "rank": rank,
           "scale": (alpha or 2 * rank) / rank, "lr": lr, "schedule": schedule, "warmup": warmup, "min_lr_ratio": min_lr_ratio,
           "weight_decay": weight_decay, "micro": micro, "micro_batch_size": micro_batch_size, "codec": codec, "ckpt_every": ckpt_every,
           "loss_scale": 1024.0, "loss_scale_max": 1024.0, "checkpointing": True, "peer_replicas": "auto", "rebalance": True, "rebalance_cost_s": 60, "streams": 3, "pad_id": int(pad),
           "expected_steps": expected_steps, "seq": seq, "sizes": sizes, "auto": auto, "layout": lay, "mlx_ok": mlx_ok,
           "model_type": model_type, "linear_attention": "linear_attention" in (getattr(text, "layer_types", None) or []),
           **sizes["torch"]}
    if guessed:
        job["mlx_guessed"] = guessed
    unknown = sorted(set(extra) - set(RUN_EXTRAS))
    if unknown:
        raise TypeError(f"unknown setting{'s' if len(unknown) > 1 else ''} for a run: {', '.join(unknown)} (the settings besides "
                        f"the named ones are {', '.join(RUN_EXTRAS)})")
    job.update(extra)
    targets = job.get("target_modules")
    if targets in ("all-linear", None):
        job["target_modules"] = None
    elif isinstance(targets, str):
        job["target_modules"] = [targets]
    elif not (isinstance(targets, (list, tuple)) and targets and all(isinstance(t, str) and t for t in targets)):
        raise ValueError(f"target_modules must be layer names such as [\"q_proj\", \"v_proj\"], or \"all-linear\", not {targets!r}")
    else:
        job["target_modules"] = list(targets)
        if mode == "lora":
            block = skeleton(config, lay["auto"]).get_submodule(lay["stack"])[0]
            names = lora_targets(block)
            if not any(targeted(n, job["target_modules"]) for n in names):
                raise ValueError(f"target_modules {job['target_modules']} names none of the linear layers in a block of {model}: "
                                 f"{', '.join(names)}")
    for key in ("checkpointing", "rebalance"):
        if not isinstance(job[key], bool):
            raise ValueError(f"{key} must be True or False, not {job[key]!r}")
    choices = {"codec": ("int8", "fp16", "fp32"), "peer_replicas": ("auto", "always", "never"), "schedule": ("constant", "linear", "cosine")}
    for key, allowed in choices.items():
        if job[key] not in allowed:
            raise ValueError(f"{key} must be one of {', '.join(allowed)}, not {job[key]!r}")
    for key in ("micro", "ckpt_every", "streams", "seq", "rank"):
        if not isinstance(job[key], int) or job[key] < 1:
            raise ValueError(f"{key} must be a whole number of at least 1, not {job[key]!r}")
    if job["micro_batch_size"] is not None and (not isinstance(job["micro_batch_size"], int) or job["micro_batch_size"] < 1):
        raise ValueError(f"micro_batch_size must be a whole number of at least 1, not {job['micro_batch_size']!r}")
    return job


def load_rows(path: str) -> list:
    if path.endswith((".jsonl", ".json")):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()] if path.endswith(".jsonl") else json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{path}: a .json file must hold a list of rows (strings, or objects with 'text' or 'messages')")
        out = []
        for row in rows:
            if isinstance(row, str):
                out.append(row)
            elif "messages" in row:
                out.append(row["messages"])
            elif "text" in row:
                out.append(row["text"])
            else:
                raise ValueError(f"{path}: every row needs a 'text' or 'messages' field")
        return out
    if not path.endswith((".txt", ".md")):
        raise ValueError(f"{path}: give the data as .jsonl or .json rows, or as a .txt or .md file of plain text")
    with open(path, encoding="utf-8") as f:
        return [f.read()]


def _render(tokenizer, messages: list[dict], prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=prompt)


def assistant_spans(tokenizer, messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    full = _render(tokenizer, messages)
    spans, cursor = [], 0
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before, upto = _render(tokenizer, messages[:i], prompt=True), _render(tokenizer, messages[:i + 1])
        if upto.startswith(before) and full.startswith(upto):
            spans.append((len(before), len(upto)))
            cursor = len(upto)
        else:
            at = full.find(message["content"], cursor)
            if at < 0:
                raise ValueError("could not locate an assistant reply in the chat template output")
            spans.append((at, at + len(message["content"])))
            cursor = at + len(message["content"])
    return full, spans


def encode_row(tokenizer, row, assistant_only: bool = True) -> tuple[list[int], list[int]]:
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if isinstance(row, str):
        ids = list(tokenizer(row)["input_ids"])
        if not ids or ids[-1] != eos:
            ids.append(eos)
        return ids, list(ids)
    text, spans = assistant_spans(tokenizer, row)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    if not assistant_only:
        return ids, list(ids)
    labels = [t if any(a <= start < b for a, b in spans) else -100 for t, (start, _) in zip(ids, enc["offset_mapping"])]
    return ids, labels

import atexit
import json
import math
import os
import threading
import time
import webbrowser

import numpy as np

from . import __version__, selfcheck, trainers
from .blocks import BETAS, EPS, load_ends
from .control import ControlServer, RemoteError, RemotePool, find_local
from .driver import Ends, PoolChain
from .export import BLOCKS_FILE, ENDS_FILE, add_saved_modules, write_ends
from .job import build_job
from .pool import OPTIMIZER_FILE, Pool, PoolError, scheduled_lr
from .worker import Worker, cuda_problem, device_identity, native_bf16, stored_identity

STATE_DIR = os.path.join(os.path.expanduser("~"), ".layerpool")


def _rows(input_ids) -> list[list[int]]:
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], int):
        input_ids = [input_ids]
    return [list(map(int, r)) for r in input_ids]


def pad_rows(rows: list[list[int]], fill: int, length: int | None = None) -> np.ndarray:
    length = length or max(len(r) for r in rows)
    out = np.full((len(rows), length), fill, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r[:length]
    return out


LOCAL_FILE = "layerpool_local.safetensors"
LOCAL_OPTIMIZER_FILE = "layerpool_local_optimizer.pt"


class Result(dict):
    __getattr__ = dict.get


class Run:
    """One training run on the pool. `run.model` is the transformers model on this machine; its repeated blocks live on the
    devices. Your loop calls forward_backward / optim_step (or use the model with any transformers trainer)."""

    def __init__(self, pool, run_id: str, job: dict, tokenizer=None, ends=None):
        self.pool, self.id, self.job, self.tokenizer, self.ends = pool, run_id, job, tokenizer, ends
        self.model = ends.model if ends is not None else None
        self._ended = False
        atexit.register(self._end_at_exit)
        self.step = 0
        self.metrics: list[dict] = []
        self._loss_accum = None
        self._loss_parts: list[tuple[float, int | None]] = []
        self._local_opt = None

    def _add_loss(self, value: float, tokens: int | None = None):
        """One forward pass's loss. A loss averaged over its own tokens (tokens given) counts by its tokens, so the step shows the
        average over the accumulated batches; a loss already divided by the whole step's count (tokens None) adds up."""
        self._loss_parts.append((value, tokens))
        if all(t is not None for _, t in self._loss_parts):
            self._loss_accum = sum(v * t for v, t in self._loss_parts) / max(1, sum(t for _, t in self._loss_parts))
        else:
            self._loss_accum = sum(v for v, _ in self._loss_parts)

    def _prepare(self, input_ids, labels=None, weights=None):
        pad = self.job.get("pad_id", 0)
        if isinstance(input_ids, np.ndarray) and input_ids.ndim == 2:
            ids = input_ids.astype(np.int64)
        elif hasattr(input_ids, "detach") and input_ids.dim() == 2:
            ids = input_ids.detach().cpu().numpy().astype(np.int64)
        else:
            rows = _rows(input_ids)
            ids = pad_rows(rows, pad)
            if labels is None:
                labels = pad_rows(rows, -100)
                for i, r in enumerate(rows):
                    labels[i, :len(r)] = r
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()
        targets = ids.copy() if labels is None else (labels.astype(np.int64) if isinstance(labels, np.ndarray) else pad_rows(_rows(labels), -100, ids.shape[1]))
        if weights is None:
            w = (targets[:, 1:] != -100).astype(np.float32)
        else:
            w = np.asarray(weights, np.float32)
            w = w[:, 1:] if w.shape[1] == ids.shape[1] else w
        return ids, targets, w

    def _logits(self, ids: np.ndarray, micro_rows: int | None, **inputs):
        import torch
        old = self.job.get("micro_batch_size")
        if micro_rows:
            self.job["micro_batch_size"] = micro_rows
        try:
            device = self.model.device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            out = self.model(input_ids=torch.tensor(ids, device=device), **inputs)
        finally:
            self.job["micro_batch_size"] = old
        return out.logits

    @staticmethod
    def _token_logprobs(logits, targets):
        import torch
        from .losses import token_logprobs
        t = torch.tensor(targets[:, 1:], device=logits.device).clamp(min=0)
        return token_logprobs(logits[:, :-1], t)

    def forward_backward(self, input_ids, labels=None, weights=None, loss="ce", normalize="mean", micro_rows: int | None = None, data=None,
                         **inputs) -> Result:
        """Add this batch's gradients. `labels` use -100 for tokens that should not be learned. `loss` is "ce" (cross-entropy on the
        labelled tokens) or a function loss(logprobs, mask, data) -> scalar torch tensor computed from the per-token log-probabilities.
        `normalize` is "mean" (over the batch's labelled tokens), "sum", or a number to divide by. Extra keyword arguments go to the
        model (pixel_values for an image model, for example)."""
        import torch
        t0 = time.time()
        ids, targets, w = self._prepare(input_ids, labels, weights)
        tokens = int((w > 0).sum())
        self.model.train()
        logits = self._logits(ids, micro_rows, **inputs)
        lp = self._token_logprobs(logits, targets).cpu()
        mask = torch.tensor(w)
        if callable(loss):
            value = loss(lp, mask, data)
        elif loss == "ce":
            divisor = float(w.sum()) if normalize == "mean" else 1.0 if normalize == "sum" else float(normalize)
            value = -(lp * mask).sum() / max(divisor, 1e-9)
        else:
            raise ValueError(f"unknown loss {loss!r}; pass 'ce' or a function of (logprobs, mask, data)")
        value.backward()
        number = float(value.detach())
        self._add_loss(number, tokens if loss == "ce" and normalize == "mean" else None)
        self.pool.add_loss(number)
        return Result(loss=number, tokens=tokens, seconds=round(time.time() - t0, 3))

    def forward(self, input_ids, labels=None, adapter: bool = True, micro_rows: int | None = None, metrics: bool = False, **inputs):
        """Per-token log-probabilities of the next tokens, shape (rows, tokens - 1). adapter=False gives the base model (LoRA runs).
        metrics=True also returns the entropy of each next-token distribution and whether the target was the most likely token."""
        import torch
        ids, targets, _ = self._prepare(input_ids, labels)
        self.ends.adapter = adapter
        try:
            with torch.no_grad():
                logits = self._logits(ids, micro_rows, **inputs)
        finally:
            self.ends.adapter = True
        if not metrics:
            return self._token_logprobs(logits, targets).cpu().numpy()
        from .losses import token_stats
        lp, entropy, top1 = token_stats(logits[:, :-1], torch.tensor(targets[:, 1:], device=logits.device).clamp(min=0))
        return Result(logprobs=lp.cpu().numpy(), entropy=entropy.cpu().numpy(), top1=top1.cpu().numpy())

    def _new_local_opt(self, lr: float):
        """AdamW for the trained parts on this machine: matrices decay, norms and biases do not (as in the Trainer)."""
        import torch
        params = self._local_params()
        groups = [{"params": [p for p in params if p.dim() > 1], "weight_decay": self.job.get("weight_decay", 0.0)},
                  {"params": [p for p in params if p.dim() <= 1], "weight_decay": 0.0}]
        return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, betas=BETAS, eps=EPS)

    def _local_params(self):
        return [p for n, p in self.model.named_parameters() if p.requires_grad and n != "layerpool_anchor"] if self.model is not None else []

    def optim_step(self, lr: float | None = None, max_grad_norm: float | None = 1.0, loss: float | None = None) -> Result:
        """One optimizer step on every device (and on this machine's part of the model in full fine-tuning), with the gradient
        norm clipped across all of them."""
        lr = self.job["lr"] if lr is None else lr
        self.ends.chain.drop_pending()
        local = [p for p in self._local_params() if p.grad is not None]
        local_sq = float(sum(p.grad.float().pow(2).sum() for p in local)) if local else 0.0
        remote = float(self.ends.remote_norm) ** 2
        total = math.sqrt(local_sq + remote)
        clip = min(1.0, max_grad_norm / (total + 1e-6)) if max_grad_norm else 1.0
        if not math.isfinite(total):
            clip = float("nan")
        record = self.pool.optim_step(lr, None, self._loss_accum if loss is None else loss, clip=clip,
                                      weight_decay=self.job.get("weight_decay"))
        if local:
            if record.get("skipped"):
                for p in local:
                    p.grad = None
            else:
                if self._local_opt is None:
                    self._local_opt = self._new_local_opt(lr)
                for group in self._local_opt.param_groups:
                    group["lr"] = lr
                for p in local:
                    p.grad.mul_(clip)
                self._local_opt.step()
                self._local_opt.zero_grad(set_to_none=True)
        self.ends.anchor.grad = None
        self.ends.remote_norm = 0.0
        self._loss_accum = None
        self._loss_parts = []
        self.step = record["step"] + 1
        return Result(record)

    def log(self, **metrics):
        self.metrics.append({"step": self.step, "time": time.time(), **metrics})
        self.pool.emit("metric", self.metrics[-1])

    def save(self, path: str, dtype: str = "bfloat16", optimizer: bool = False) -> str:
        """Write the result: a PEFT adapter folder for LoRA runs, a full Hugging Face model folder for full fine-tuning.
        optimizer=True also keeps what resuming needs exactly: the devices' AdamW state and this machine's trained weights in
        float32 (load() reads them back)."""
        self.ends.chain.drop_pending()
        path = os.path.abspath(path)
        self.pool.export(path, dtype, self.metrics, optimizer=optimizer)
        if self.job.get("mode") == "full":
            write_ends(self.model, path, self.job, dtype)
        elif self.job.get("modules_to_save"):
            add_saved_modules(self.model, path, self.job["modules_to_save"])
        if optimizer:
            from safetensors.torch import save_file
            mine = {n: p.detach().float().cpu().contiguous() for n, p in self.model.named_parameters()
                    if p.requires_grad and n != "layerpool_anchor"}
            save_file(mine, os.path.join(path, LOCAL_FILE))
            if self._local_opt is not None:
                import torch
                torch.save(self._local_opt.state_dict(), os.path.join(path, LOCAL_OPTIMIZER_FILE))
        return path

    def load(self, path: str):
        """Continue from a folder this library saved (save(), save_pretrained or a Trainer checkpoint): a PEFT adapter for LoRA
        runs, a model folder for full fine-tuning. Before the first step it is where training starts; later it replaces what the devices
        have trained so far (as the Trainer does for load_best_model_at_end)."""
        import torch
        from safetensors.numpy import load_file
        from safetensors.torch import load_file as load_torch
        self.ends.chain.drop_pending()
        stack = self.job["layout"]["stack"] + "."
        where = lambda name: os.path.join(path, name)
        remote = load_file(where(OPTIMIZER_FILE)) if os.path.exists(where(OPTIMIZER_FILE)) else None
        local = {}
        if self.job.get("mode") == "full":
            if remote is None:
                remote = {k[len(stack):]: v.float().numpy() for k, v in load_torch(where(BLOCKS_FILE)).items()}
            local = load_torch(where(ENDS_FILE))
        else:
            if not os.path.exists(where("adapter_model.safetensors")):
                raise FileNotFoundError(f"{path} has no adapter_model.safetensors: it is not a LoRA adapter saved by layerpool or PEFT")
            self._check_adapter(path)
            prefix = "base_model.model."
            found = {}
            for key, value in load_file(where("adapter_model.safetensors")).items():
                name = key[len(prefix):] if key.startswith(prefix) else key
                if name.startswith(stack) and ".lora_" in name:
                    module, which = name[len(stack):].split(".lora_")
                    found[f"{module}.{which.split('.')[0]}"] = value
                else:
                    local[name] = torch.from_numpy(value)
            remote = remote if remote is not None else found
        if os.path.exists(where(LOCAL_FILE)):
            local = {**local, **load_torch(where(LOCAL_FILE))}
        if local:
            names = self.model.state_dict(keep_vars=True)
            unknown = [k for k in local if k not in names]
            if unknown:
                raise ValueError(f"{path} has weights this model does not: {unknown[:5]}")
            with torch.no_grad():
                for k, v in local.items():
                    names[k].copy_(v.to(names[k].dtype))
        if os.path.exists(where(LOCAL_OPTIMIZER_FILE)) and self._local_params():
            self._local_opt = self._new_local_opt(self.job["lr"])
            self._local_opt.load_state_dict(torch.load(where(LOCAL_OPTIMIZER_FILE), weights_only=True))
        self.pool.load_weights(remote)

    def _check_adapter(self, path: str):
        """The adapter must have this run's rank and scale, or its updates would come out at the wrong size."""
        config_path = os.path.join(path, "adapter_config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        rank, alpha = config.get("r"), config.get("lora_alpha")
        if rank is not None and rank != self.job["rank"]:
            raise ValueError(f"{path} is a rank-{rank} adapter and this run has rank {self.job['rank']}: start the run with lora_r={rank}")
        if rank and alpha is not None:
            scale = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)
            if abs(scale - self.job["scale"]) > 1e-6 * max(1.0, scale):
                raise ValueError(f"{path} was trained with lora_alpha={alpha} (scale {scale:g}) and this run has scale "
                                 f"{self.job['scale']:g}: start the run with lora_alpha={alpha / rank * self.job['rank']:g}")

    def end(self, state: str = "finished"):
        if self._ended:
            return
        self._ended = True
        atexit.unregister(self._end_at_exit)
        if self.ends is not None:
            self.ends.chain.drop_pending()
        self.pool.end_run(state, self.id)

    def _end_at_exit(self):
        try:
            self.end("finished" if getattr(self, "trained", False) else "stopped")
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        self.end("failed" if kind else "finished")

    def schedule(self, step: int, total: int) -> float:
        return scheduled_lr(self.job["lr"], step, total, self.job.get("warmup", 0), self.job.get("schedule", "constant"), self.job.get("min_lr_ratio", 0.1))

    def train(self, data, method: str = "sft", steps: int | None = None, epochs: float = 1.0, batch_size: int = 8, seq: int | None = None,
              max_grad_norm: float | None = 1.0, beta: float | None = None, on_step=None, seed: int = 0, **kw) -> list[dict]:
        """Built-in loops. method: "sft" (chat or text rows, learns assistant replies), "cpt" (raw text, every token),
        "dpo" / "ipo" / "simpo" / "orpo" (rows with prompt, chosen, rejected). data: rows, a dataset, a file, or the name of a
        dataset on Hugging Face (with split= and subset=)."""
        return trainers.train(self, data, method=method, steps=steps, epochs=epochs, batch_size=batch_size, seq=seq, max_grad_norm=max_grad_norm,
                              beta=beta, on_step=on_step, seed=seed, **kw)


def _choose_attention(model, wanted: str | None):
    """The attention code of the part on this machine: the one asked for, or, on an Apple GPU, the plain one when the model has
    attention dropout (PyTorch's fused attention on Apple GPUs cannot apply dropout in training)."""
    import torch
    if wanted is None:
        text = model.config.get_text_config()
        dropout = max(float(getattr(c, name, 0.0) or 0.0) for c in (model.config, text)
                      for name in ("attention_dropout", "attention_probs_dropout_prob", "dropout_rate", "attn_pdrop"))
        if not (dropout > 0 and torch.backends.mps.is_available()):
            return
        wanted = "eager"
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(wanted)
    else:
        model.config._attn_implementation = wanted
    if wanted == "eager":
        for mod in model.modules():
            config = getattr(mod, "config", None)
            if getattr(config, "_attn_implementation", None) == "sdpa":
                config._attn_implementation = "eager"


def _this_machine() -> list[str]:
    """The ids this machine's own devices joined with (layerpool up, or layerpool join), without making new ones."""
    found = (stored_identity(os.path.join(STATE_DIR, name)) for name in ("local_device_id", "device_id"))
    return [value for value in found if value]


def _inputs_as(say, convert: bool):
    """A pre-hook that gives the model its floating inputs (audio features, pixels) in the model's own dtype, so code written for
    a float32 model keeps working when this machine holds the model in half precision. A half-precision model that is still on
    the CPU when it first runs (a hand-written loop that did not move it to a GPU) goes to float32 first: a CPU is hundreds of
    times slower in bfloat16."""
    import torch

    def hook(module, args, kwargs):
        if kwargs.get("output_hidden_states") or kwargs.get("output_attentions"):
            from .blocks import NO_INNER_OUTPUTS
            raise ValueError(NO_INNER_OUTPUTS)
        p = next((p for n, p in module.named_parameters() if p.is_floating_point() and n != "layerpool_anchor"), None)
        if p is None:
            return None
        if convert and p.device.type == "cpu" and p.dtype in (torch.bfloat16, torch.float16):
            module.float()
            say("this machine's part of the model runs on the CPU, so in float32; move the model to the GPU (model.to(\"cuda\") "
                "or \"mps\") to run it there in bfloat16")
            p = next(q for n, q in module.named_parameters() if q.is_floating_point() and n != "layerpool_anchor")
        if p.dtype == torch.float32:
            return None

        def cast(v):
            return v.to(p.dtype) if torch.is_tensor(v) and v.dtype == torch.float32 else v
        return tuple(cast(a) for a in args), {k: cast(v) for k, v in kwargs.items()}
    return hook


def best_device() -> str:
    """Where this machine runs the parts around the blocks: its NVIDIA/AMD GPU if PyTorch can train on it, else the Apple GPU,
    else the CPU."""
    import torch
    if torch.cuda.is_available() and cuda_problem() is None:
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ends_gb(repo: str, lay: dict, dtype) -> float:
    """Memory the parts around the blocks take on this machine, from the checkpoint's tensor sizes, so the split can count it
    before they are loaded (they load while the devices load their blocks)."""
    import torch
    from .blocks import block_keys
    from .job import tensor_bytes
    try:
        sizes = tensor_bytes(repo)
        inside = block_keys(sizes, lay["stack"], lay["n"])
        held = sum(v for k, v in sizes.items() if k not in inside)
    except Exception:
        return 0.5
    wide = torch.tensor([], dtype=dtype).element_size() if isinstance(dtype, torch.dtype) else 2
    return round(held * max(1.0, wide / 2) / 1e9 + 0.5, 2)


def _local_dtype(choice, mode: str, codec: str, device: str | None = None):
    """With "auto", bfloat16 only where this machine's part will run on a GPU that is fast at it: a CPU is far slower in bfloat16
    than in float32."""
    import torch
    if isinstance(choice, torch.dtype):
        return choice
    if choice in ("float32", "bfloat16", "float16"):
        return getattr(torch, choice)
    if choice != "auto":
        raise ValueError(f"dtype must be \"auto\", \"float32\", \"bfloat16\" or \"float16\", not {choice!r}")
    if mode == "full" or codec == "fp32" or device == "cpu":
        return torch.float32
    if device == "mps":
        return torch.bfloat16
    if torch.cuda.is_available() and cuda_problem() is None:
        return torch.bfloat16 if native_bf16() else torch.float32
    if torch.backends.mps.is_available():
        return torch.bfloat16
    return torch.float32


class Client:
    """A handle on a pool: the in-process one started by start(), or one already running from `layerpool up`."""

    def __init__(self, pool: Pool | None = None, remote=None, local_worker=None):
        self.pool = pool
        self.remote = remote
        self.local_worker = local_worker

    @property
    def _p(self):
        return self.pool or self.remote

    def devices(self) -> list[dict]:
        return self._p.devices()

    def status(self) -> dict:
        return self._p.status()

    def invite(self, reusable: bool = False, hours: float = 24) -> dict:
        """A join command for a new device; reusable lets several devices join with it."""
        return self._p.invite(reusable=reusable, ttl=hours * 3600)

    def _device_id(self, device: str) -> str:
        found = [d["id"] for d in self.devices() if device in (d["id"], d["name"]) and d["status"] != "removed"]
        if not found:
            raise ValueError(f"no device {device!r} in this pool")
        return found[0]

    def set_device(self, device: str, **settings) -> dict:
        """Change a device's settings, by its id or name: name, cap_gb, enabled, layers (8, "30%", "4-10"), gpus ([0, 2]),
        gpu_caps ({0: 6, 1: 10} in GB), order (its place in the chain) and ram_cap (GB of RAM a machine with GPUs may also use for
        layers its CPU runs); None or "auto" clears one."""
        if "gpu_caps" in settings and isinstance(settings["gpu_caps"], dict):
            settings["gpu_caps"] = {str(k): v for k, v in settings["gpu_caps"].items()}
        return self._p.set_device(self._device_id(device), **settings)

    def pause(self, device: str):
        """Stop using a device; its layers go to the others."""
        self._p.drain(self._device_id(device))

    def resume(self, device: str):
        self._p.set_device(self._device_id(device), enabled=True)

    def remove(self, device: str):
        """Take a device out of the pool; it needs a new invite to come back."""
        self._p.remove(self._device_id(device))

    def waiting(self) -> list[dict]:
        """Devices that asked to join without a code, each with the code it shows."""
        return self._p.waiting()

    def approve(self, code: str, yes: bool = True) -> str:
        """Let a waiting device in (yes=False turns it away)."""
        return self._p.answer(code, yes)

    def end_run(self):
        """End the current run at once, from anywhere (its script gets an error at its next call)."""
        run = self.status().get("run")
        if run:
            self._p.end_run("stopped", run)

    def plan(self, model: str, mlx_model: str | None = None, mode: str = "lora", rank: int = 16, seq: int = 512, micro: int = 8,
             micro_batch_size: int | None = None, auto: str = "AutoModelForCausalLM", **settings) -> dict:
        """Whether a run of this model fits the devices here now and how it would be split, before starting it: {"plan": [{"name",
        "range", "gb"}, ...] or None, "why": the reason when it does not fit, "layers", "layer_gb"}. The modules around the
        blocks (embeddings, output head) stay on the machine that runs the training code and are not counted here."""
        job = build_job(model, mlx_model=mlx_model, mode=mode, rank=rank, seq=seq, micro=micro, micro_batch_size=micro_batch_size,
                        auto=auto, **settings)
        return self._p.plan_for(job)

    def set_run(self, rebalance: bool | None = None, resplit: bool = False, expected_steps: int | None = None) -> dict:
        """Change the running run: rebalance=False keeps the current split, resplit=True splits the layers again now,
        expected_steps tells rebalancing how long the run is."""
        return self._p.set_run(rebalance=rebalance, resplit=resplit, expected_steps=expected_steps)

    def wait_for_devices(self, count: int = 1, timeout: float = 600) -> list[dict]:
        deadline, live = time.time() + timeout, []
        while True:
            live = [d for d in self.devices() if d["status"] in ("idle", "training")]
            if len(live) >= count:
                return live
            if time.time() >= deadline:
                break
            time.sleep(0.5)
        raise TimeoutError(f"only {len(live)} of {count} devices joined")

    def run(self, model: str, mlx_model: str | None = None, mode: str = "lora", rank: int = 16, alpha: float | None = None, lr: float = 1e-4,
            schedule: str = "cosine", warmup: int = 0, min_lr_ratio: float = 0.1, weight_decay: float = 0.0, micro: int = 8,
            micro_batch_size: int | None = None, codec: str = "fp16", ckpt_every: int = 4, random_init: bool = False, expected_steps: int | None = None,
            seq: int = 512, name: str | None = None, timeout: float = 1800, modules_to_save: list[str] | None = None,
            config: dict | None = None, dtype: str = "auto", check: bool = True, attn_implementation: str | None = None,
            device: str | None = "auto", **extra) -> Run:
        """Start a run. mode: "lora" (also QLoRA on a 4-bit model), "full" (every weight), random_init=True trains from scratch.
        modules_to_save: modules around the blocks that a LoRA run trains in full, as in PEFT (a head the checkpoint lacks, such as
        a new classifier, is always trained). config: settings for the model's config, such as {"num_labels": 3}.
        dtype: how this machine holds the parts around the blocks (embeddings, output head, encoders): "float32", "bfloat16",
        "float16", or "auto" (bfloat16 when nothing here is trained and a GPU or Apple GPU is present, which halves this machine's
        memory; float32 otherwise, and always with codec="fp32").
        check: first split a small copy of the model's architecture and compare it with the whole copy (once per architecture),
        so a model layerpool cannot split correctly stops here instead of training wrong.
        device: where this machine runs its part ("auto": its GPU, the Apple GPU, or the CPU; "cuda", "mps", "cpu"); None leaves
        the model on the CPU for you or a Trainer to move."""
        import torch
        place = best_device() if device == "auto" else device
        local = _local_dtype(dtype, mode, codec, place)
        status = self._p.status()
        going = status.get("run")
        quiet = status.get("script_quiet_s")
        if going and not (quiet is not None and quiet > status.get("abandon_after_s", 20)):
            raise PoolError(f"run {going} is still going on this pool: wait for it, or end it with `layerpool end` (or in the dashboard)")
        job = build_job(model, mlx_model=mlx_model, mode=mode, rank=rank, alpha=alpha, lr=lr, schedule=schedule, warmup=warmup, min_lr_ratio=min_lr_ratio,
                        weight_decay=weight_decay, micro=micro, micro_batch_size=micro_batch_size, codec=codec, ckpt_every=ckpt_every,
                        random_init=random_init, expected_steps=expected_steps, seq=seq, **extra)
        say = getattr(self._p, "event", None) or (lambda text: print(f"[layerpool] {text}", flush=True))
        if job.get("mlx_guessed"):
            say(f"Macs train this 4-bit model with {job['mlx_guessed']} (found on the Hub by its name); pass mlx_model= to choose another")
        if check:
            notes = selfcheck.check(model, job["auto"], log=say)
            if "place" in notes and job["kind"] == "q4":
                raise ValueError(f"{model} cannot be trained in 4-bit by layerpool yet: its blocks compute constants from where they "
                                 "sit in the model, and the 4-bit path cannot rebuild them in place. Train the 16-bit model instead")
            if "whole_batches" in notes:
                job["micro"], job["micro_batch_size"] = 1, None
        if modules_to_save and dtype == "auto":
            local = torch.float32
        job["driver"] = {"identities": _this_machine(), "gb": _ends_gb(model, job["layout"], local)}
        job["config"] = dict(config or {})
        run_id = self._p.start_run(job, name)
        try:
            here, local = self._load_ends(model, job, local, mode, modules_to_save, dtype, random_init, config, attn_implementation, say)
            self._wait_ready(timeout, say)
        except BaseException:
            self._p.end_run("failed", run_id)
            raise
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception:
            pass
        ends = Ends(here, job["layout"], PoolChain(self._p, job))
        if place:
            here.to(place)
        return Run(self._p, run_id, job, tokenizer, ends)

    def _load_ends(self, model, job, local, mode, modules_to_save, dtype, random_init, config, attn_implementation, say):
        """This machine's part of the model (embeddings, output head, encoders), loaded while the devices load theirs."""
        import torch
        here = load_ends(model, job["layout"], local, random_init=random_init, log=say, overrides=config)
        here.config._name_or_path = model
        here.name_or_path = model
        _choose_attention(here, attn_implementation)
        stack = job["layout"]["stack"]
        saved = {n.rsplit(".", 1)[0] for n in here.__dict__.get("_layerpool_new", [])}
        names = [n for n, _ in here.named_modules() if n]
        for wanted in modules_to_save or []:
            found = [n for n in names if n == wanted or n.endswith("." + wanted)]
            if not found:
                raise ValueError(f"modules_to_save: this model has no module named {wanted!r}")
            inside = [n for n in found if n == stack or n.startswith(stack + ".")]
            if inside:
                raise ValueError(f"modules_to_save: {inside[0]} is part of the blocks; it is for the modules around them")
            saved.update(found)
        saved = sorted(saved)
        if saved and dtype == "auto" and local != torch.float32:
            here, local = here.float(), torch.float32
        for n, p in here.named_parameters():
            p.requires_grad_(mode == "full" or any(n.startswith(m + ".") for m in saved))
        job["modules_to_save"] = saved if mode != "full" else []
        job["local_dtype"] = str(local).replace("torch.", "")
        here.register_forward_pre_hook(_inputs_as(say, dtype == "auto"), with_kwargs=True)
        return here, local

    def _wait_ready(self, timeout: float, say):
        """Wait for the devices to load their layers, saying here what the pool does meanwhile (devices downloading, joining) and
        when no device has joined yet, instead of waiting in silence."""
        deadline, told_empty = time.time() + timeout, 0.0
        shown = {tuple(e) for e in (self._p.status().get("events") or [])} if self.remote else set()
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError("the devices did not get ready in time; check the dashboard or `layerpool devices`")
            try:
                self._p.wait_ready(min(15.0, left))
                return
            except (TimeoutError, RemoteError) as e:
                if isinstance(e, RemoteError) and "did not get ready in time" not in str(e):
                    raise
            status = self._p.status()
            if self.remote:
                for event in status.get("events") or []:
                    if tuple(event) not in shown:
                        shown.add(tuple(event))
                        print(f"[layerpool] {event[1]}", flush=True)
            if not status.get("devices") and time.time() - told_empty > 60:
                told_empty = time.time()
                say("waiting for devices: none has joined this pool yet. Add one from the dashboard or with `layerpool invite`")

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self.local_worker is not None:
            self.local_worker.alive = False
        if getattr(self, "control", None) is not None:
            self.control.close()
        if self.pool is not None:
            self.pool.close()


def start(port: int = 0, tunnel: bool = False, work: bool = True, cap_gb: float | None = None, name: str | None = None, backend: str = "auto",
          cpu: bool = False, heartbeat_timeout: float = 120.0, log=None, control: bool = True, open_browser: bool = False,
          layers=None, gpus=None, gpu_caps=None, order: int | None = None, ram_cap: float | None = None) -> Client:
    """Start a pool inside this Python process. This machine trains too unless work=False; the device settings (cap_gb,
    layers, gpus, gpu_caps, order, ram_cap) are those of `layerpool join`."""
    pool = Pool(port=port, tunnel=tunnel, heartbeat_timeout=heartbeat_timeout, log=log)
    local = None
    if work:
        local = Worker(f"http://127.0.0.1:{pool.server.port}", pool.token, name, cap_gb, tunnel=tunnel, backend=backend, cpu=cpu,
                       identity=device_identity(os.path.join(os.path.expanduser("~"), ".layerpool", "local_device_id")),
                       log=(lambda *a: None) if log is not None else None, layers=layers, gpus=gpus, gpu_caps=gpu_caps, order=order, ram_cap=ram_cap)
        threading.Thread(target=local.run, daemon=True).start()
    client = Client(pool=pool, local_worker=local)
    if control:
        client.control = ControlServer(pool)
        client.control.publish()
        if open_browser:
            webbrowser.open(client.control.login_url())
    atexit.register(client.close)
    return client


def connect(url: str | None = None, token: str | None = None, start_if_missing: bool = True, **start_kwargs) -> Client:
    """Attach to a running pool (from `layerpool up`) or start one here. Looks at LAYERPOOL_URL / LAYERPOOL_TOKEN, then ~/.layerpool/pool.json."""
    url = url or os.environ.get("LAYERPOOL_URL")
    token = token or os.environ.get("LAYERPOOL_TOKEN")
    if not url:
        found = find_local()
        if found:
            url, token = found["url"], found["token"]
    if url:
        remote = RemotePool(url, token)
        try:
            theirs = str(remote.status().get("version", ""))
        except RemoteError as e:
            raise PoolError(f"the pool at {url} refused this script ({e}): check LAYERPOOL_TOKEN, or unset LAYERPOOL_URL to use "
                            "the pool `layerpool up` started on this computer") from None
        except OSError:
            raise PoolError(f"cannot reach a pool at {url}: is `layerpool up` running there?") from None
        if theirs and theirs.split(".")[:2] != __version__.split(".")[:2]:
            raise RuntimeError(f"the pool runs layerpool {theirs} and this script {__version__}: install the same version in both "
                               f"(pip install layerpool=={theirs})")
        return Client(remote=remote)
    if not start_if_missing:
        raise PoolError("no running pool found; start one with `layerpool up`")
    print("[layerpool] no pool is running on this computer, so this script starts its own and trains here alone. To add other "
          "devices, run `layerpool up` in a terminal first.", flush=True)
    return start(**start_kwargs)

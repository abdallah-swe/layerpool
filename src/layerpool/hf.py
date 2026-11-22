"""Use a layerpool run as a Hugging Face model, so transformers.Trainer, TRL trainers and hand-written loops work unchanged.

    model = layerpool.hf.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", lora_r=16)
    trainer = transformers.Trainer(model=model, args=TrainingArguments(...), train_dataset=ds, data_collator=...)
    trainer.train()
    model.save_pretrained("out")

The model you get is the real transformers model: its embeddings, output head and any image or audio encoder are on this machine,
and its repeated blocks run on the devices. Every AutoModel class works (layerpool.hf.AutoModelForImageTextToText, AutoModel, ...).
The model shows torch one stand-in parameter whose gradient equals the gradient norm on the devices, so clip_grad_norm_ works
unchanged, and every optimizer.step() is forwarded to the devices with the learning rate the scheduler set and the clipping factor
torch applied."""

import inspect
import os
import warnings

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

from .api import connect
from .blocks import load_config

_runs: dict = {}


def _is_ours(model) -> bool:
    return getattr(model, "layerpool_run", None) is not None


def _step_hook(optimizer, args, kwargs):
    for group in optimizer.param_groups:
        for p in group["params"]:
            run = _runs.get(id(p))
            if run is not None:
                _optimizer_step(run, optimizer, group)


def _optimizer_step(run, optimizer, group):
    ends = run.ends
    ends.chain.drop_pending()
    if not getattr(run, "_warned_optimizer", False) and not isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
        run._warned_optimizer = True
        warnings.warn(f"layerpool trains the devices with AdamW; {type(optimizer).__name__} only sets the learning rate and weight decay")
    norm = ends.remote_norm
    current = float(ends.anchor.grad.item()) if ends.anchor.grad is not None else norm
    clip = current / norm if norm > 0 else 1.0
    run.pool.optim_step(float(group["lr"]), None, run._loss_accum, clip=min(1.0, clip), weight_decay=group.get("weight_decay"))
    ends.remote_norm = 0.0
    run._loss_accum = None
    run._loss_parts = []


register_optimizer_step_pre_hook(_step_hook)


def _tell_length(run):
    """A Trainer callback that tells the pool how many steps the Trainer will take, so rebalancing weighs a move against the
    steps left."""
    from transformers import TrainerCallback

    class TellLength(TrainerCallback):
        layerpool_run = run

        def on_train_begin(self, args, state, control, **kwargs):
            if state.max_steps and state.max_steps > 0:
                run.pool.set_run(expected_steps=state.max_steps)
            chain = run.ends.chain
            if hasattr(chain, "expect"):
                rows, pieces = chain.pieces(int(args.per_device_train_batch_size))
                chain.expect(rows, run.job.get("seq", 512), pieces)
            if args.gradient_accumulation_steps > 1 and args.per_device_train_batch_size < 2 * len(run.pool.status().get("plan") or [1]):
                warnings.warn(f"layerpool: per_device_train_batch_size={args.per_device_train_batch_size} with gradient_accumulation_steps="
                              f"{args.gradient_accumulation_steps} sends small batches one after another, so the devices mostly wait for each "
                              "other; a bigger per_device_train_batch_size with fewer accumulation steps (the same rows per step) lets "
                              "them all work at once", stacklevel=2)
    return TellLength()


def _needs(cls, names, package: str):
    """layerpool hooks into these methods; a release that renamed one gets a clear message instead of an AttributeError."""
    missing = [n for n in names if not callable(getattr(cls, n, None))]
    if missing:
        import importlib.metadata
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "?"
        raise RuntimeError(f"this {package} ({version}) has no {cls.__name__}.{', '.join(missing)}, which layerpool hooks into; "
                           f"install the {package} version layerpool was released with, or update layerpool")


def _teach_trainer():
    """transformers.Trainer saves models it knows by their state_dict; a layerpool model saves its weights from the devices instead."""
    try:
        from transformers import Trainer
    except ImportError:
        return
    if Trainer.__dict__.get("_layerpool", False):
        return
    _needs(Trainer, ("_save", "_load_from_checkpoint", "_wrap_model", "_load_best_model", "train"), "transformers")
    save, load = Trainer._save, Trainer._load_from_checkpoint

    def _save(self, output_dir=None, state_dict=None):
        model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        if _is_ours(model):
            output_dir = output_dir or self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            model.layerpool_run.save(output_dir, optimizer=True)
            if getattr(self, "processing_class", None) is not None:
                self.processing_class.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            return
        return save(self, output_dir, state_dict)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = model or self.model
        if _is_ours(target):
            target.layerpool_run.load(resume_from_checkpoint)
            return
        return load(self, resume_from_checkpoint, model)
    wrap = Trainer._wrap_model

    def _wrap_model(self, model, *args, **kwargs):
        if _is_ours(model):
            return model
        return wrap(self, model, *args, **kwargs)
    train = Trainer.train

    def _train(self, *args, **kwargs):
        model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        if _is_ours(model) and not any(getattr(c, "layerpool_run", None) is model.layerpool_run for c in self.callback_handler.callbacks):
            self.add_callback(_tell_length(model.layerpool_run))
        out = train(self, *args, **kwargs)
        model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        if _is_ours(model):
            model.layerpool_run.trained = True
        return out
    best = Trainer._load_best_model

    def _load_best_model(self):
        model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        if _is_ours(model):
            if self.state.best_model_checkpoint:
                model.layerpool_run.load(self.state.best_model_checkpoint)
            return
        return best(self)
    Trainer._save, Trainer._load_from_checkpoint, Trainer._wrap_model, Trainer.train = _save, _load_from_checkpoint, _wrap_model, _train
    Trainer._load_best_model = _load_best_model
    Trainer._layerpool = True
    _teach_preference_trainers()


def _teach_preference_trainers():
    """TRL's DPO-style trainers load a second full copy of the model on this machine as the reference unless the reference
    log-probs are computed up front. The policy at the start of the run is that reference, so compute them up front."""
    try:
        import trl
    except Exception:
        return
    try:
        major, minor = (int(x) for x in trl.__version__.split(".")[:2])
        if (major, minor) < (1, 15):
            warnings.warn(f"layerpool needs trl>=1.15 for TRL trainers (you have {trl.__version__}): pip install -U trl")
    except ValueError:
        pass
    for name in ("DPOTrainer", "KTOTrainer"):
        try:
            cls = getattr(trl, name, None)
        except Exception:
            continue
        if cls is None or cls.__dict__.get("_layerpool", False):
            continue
        init = cls.__init__
        if list(inspect.signature(init).parameters)[1:4] != ["model", "ref_model", "args"]:
            warnings.warn(f"layerpool: this TRL's {name} takes other arguments than the versions layerpool knows, so it is not adjusted; "
                          f"give it a {name.replace('Trainer', 'Config')} with precompute_ref_log_probs=True")
            cls._layerpool = True
            continue

        def __init__(self, model=None, ref_model=None, args=None, *rest, _init=init, _name=name, **kwargs):
            if _is_ours(model) and ref_model is None and args is None and hasattr(trl, _name.replace("Trainer", "Config")):
                short = str(getattr(model.config, "_name_or_path", "model")).rstrip("/").split("/")[-1]
                args = getattr(trl, _name.replace("Trainer", "Config"))(output_dir=f"{short}-{_name[:-7]}", precompute_ref_log_probs=True)
            if _is_ours(model) and ref_model is None and args is not None and hasattr(args, "precompute_ref_log_probs") \
                    and not args.precompute_ref_log_probs:
                warnings.warn(f"layerpool: {_name} computes the reference log-probs before training (precompute_ref_log_probs=True) "
                              "instead of loading a second copy of the model on this machine")
                args.precompute_ref_log_probs = True
            _init(self, model, ref_model, args, *rest, **kwargs)
        cls.__init__, cls._layerpool = __init__, True


_LOADING = {"device_map", "low_cpu_mem_usage", "trust_remote_code", "revision", "cache_dir",
            "token", "use_safetensors", "local_files_only", "force_download", "max_memory", "offload_folder", "quantization_config"}
_LOADING_WARN = {
    "quantization_config": "load a checkpoint that is already 4-bit (for example an unsloth ...-bnb-4bit repo) for QLoRA",
    "trust_remote_code": "layerpool runs the model classes that ship with transformers, not code from the repo",
    "device_map": "the blocks go to the devices of the pool, and the rest stays on this machine",
    "max_memory": "set each device's memory cap in the dashboard or with `layerpool set DEVICE --cap-gb`",
    "offload_folder": "the blocks go to the devices of the pool",
    "revision": "every device downloads the model's main branch; to train another revision, download it to a folder that every "
                "device has, or push it to its own repo",
    "token": "each device downloads with its own Hugging Face login: run `hf auth login` or set HF_TOKEN on every device",
    "cache_dir": "each device uses its own Hugging Face cache (HF_HOME)",
    "local_files_only": "each device downloads what it needs unless HF_HUB_OFFLINE=1 is set on it",
}
_RUN_ARGS = {"mode", "rank", "alpha", "lr", "schedule", "warmup", "min_lr_ratio", "weight_decay", "micro", "micro_batch_size", "codec",
             "ckpt_every", "random_init", "expected_steps", "seq", "name", "timeout", "check", "dtype", "attn_implementation"}


class _Auto:
    """layerpool.hf.AutoModelFor...: the same class name as in transformers; from_pretrained starts a run on the pool."""

    def __init__(self, auto: str):
        self.auto = auto

    def from_pretrained(self, model: str, pool=None, mlx_model: str | None = None, lora_r: int = 16, lora_alpha: float | None = None,
                        full: bool = False, modules_to_save: list[str] | None = None, **kwargs):
        """kwargs that transformers' from_pretrained takes for the config (num_labels=3, id2label=...) set the model's config;
        dtype (or torch_dtype) is how this machine holds the parts around the blocks ("auto", "float32", "bfloat16", a torch
        dtype); the ones about where weights load (device_map, ...) do not apply, since the devices hold the blocks; the rest go to
        the run (micro, seq, codec, peer_replicas, ...)."""
        _teach_trainer()
        config = load_config(model)
        overrides, run_kwargs = {}, {}
        for key in ("mode", "rank", "alpha"):
            if key in kwargs:
                raise TypeError(f"from_pretrained takes {key} as {dict(mode='full=True', rank='lora_r', alpha='lora_alpha')[key]}")
        for key, value in kwargs.items():
            if key in ("dtype", "torch_dtype"):
                run_kwargs["dtype"] = value
                continue
            if key == "quantization_config" and value is not None and not getattr(config, "quantization_config", None):
                raise ValueError(f"layerpool cannot quantize {model} while loading it: for QLoRA load a checkpoint that is already 4-bit "
                                 "with bitsandbytes (for example unsloth/<model>-bnb-4bit) and leave out quantization_config")
            if key in _LOADING:
                if key in _LOADING_WARN and value not in (None, False) and key != "quantization_config":
                    warnings.warn(f"layerpool: {key} does not apply here: {_LOADING_WARN[key]}", stacklevel=2)
                continue
            if key in ("num_labels", "id2label", "label2id") or (hasattr(config, key) and key not in _RUN_ARGS):
                overrides[key] = value
            else:
                run_kwargs[key] = value
        run_kwargs.setdefault("schedule", "constant")
        run_kwargs.setdefault("device", None)
        client = pool or connect()
        run = client.run(model, mlx_model=mlx_model, mode="full" if full else "lora", rank=lora_r, alpha=lora_alpha,
                         auto=self.auto, modules_to_save=modules_to_save, config=overrides or None, **run_kwargs)
        return wrap_run(run)


def wrap_run(run):
    m = run.model
    key = id(run.ends.anchor)
    _runs[key] = run
    m.layerpool_run = run
    m.save_pretrained = lambda path, *a, **kw: run.save(path)
    m.load_adapter = lambda path, *a, **kw: run.load(path)

    def end(state: str = "finished"):
        _runs.pop(key, None)
        run.end(state)
    m.end = end
    zero_grad = m.zero_grad

    def drop_device_grads():
        """The training code threw this batch's gradients away without a step (a GradScaler skip, or zero_grad on purpose):
        the devices drop theirs too."""
        if run.ends.remote_norm:
            run.ends.chain.drop_pending()
            run.pool.optim_step(None, None, run._loss_accum, skip=True)
            run.ends.remote_norm = 0.0
            run._loss_accum, run._loss_parts = None, []

    def zero_grad_and_skip(*a, **kw):
        drop_device_grads()
        return zero_grad(*a, **kw)
    m.zero_grad = zero_grad_and_skip
    run.ends.discard = drop_device_grads
    m.gradient_checkpointing_enable = lambda *a, **kw: None
    m.gradient_checkpointing_disable = lambda *a, **kw: None
    m.activation_checkpointing_enable = m.gradient_checkpointing_enable
    m.activation_checkpointing_disable = m.gradient_checkpointing_disable

    def keep_loss(module, args, kwargs, output):
        loss = getattr(output, "loss", None)
        if loss is not None and torch.is_grad_enabled() and module.training:
            whole_step = kwargs.get("num_items_in_batch") is not None
            labels = kwargs.get("labels")
            tokens = None if whole_step or not torch.is_tensor(labels) else max(1, int((labels != -100).sum()))
            run._add_loss(float(loss.detach()), tokens)
    m.register_forward_hook(keep_loss, with_kwargs=True)
    return m


AutoModelForCausalLM = _Auto("AutoModelForCausalLM")


def __getattr__(name: str):
    if name.startswith("AutoModel"):
        return _Auto(name)
    raise AttributeError(name)

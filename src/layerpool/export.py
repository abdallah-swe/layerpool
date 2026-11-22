import json
import os
import re

import numpy as np

from .blocks import load_config


def _split(key: str) -> tuple[int, str, str]:
    block, rest = key.split(".", 1)
    module, which = rest.rsplit(".", 1)
    return int(block), module, which


_TASKS = {"AutoModelForCausalLM": "CAUSAL_LM", "AutoModelForSequenceClassification": "SEQ_CLS", "AutoModelForTokenClassification": "TOKEN_CLS",
          "AutoModelForSeq2SeqLM": "SEQ_2_SEQ_LM", "AutoModelForQuestionAnswering": "QUESTION_ANS", "AutoModel": "FEATURE_EXTRACTION"}


def export_peft(adapter: dict[str, np.ndarray], out_dir: str, job: dict) -> str:
    """A PEFT LoRA adapter folder: load it with peft.PeftModel.from_pretrained(base_model, out_dir)."""
    from safetensors.numpy import save_file
    stack, rank, scale = job["layout"]["stack"], job["rank"], job["scale"]
    os.makedirs(out_dir, exist_ok=True)
    tensors, modules = {}, set()
    for key, value in adapter.items():
        i, module, which = _split(key)
        modules.add(module)
        tensors[f"base_model.model.{stack}.{i}.{module}.lora_{which}.weight"] = np.ascontiguousarray(value, dtype=np.float32)
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    pattern = rf"^{re.escape(stack)}\.\d+\.({'|'.join(re.escape(m) for m in sorted(modules))})$"
    config = {"peft_type": "LORA", "task_type": _TASKS.get(job.get("auto")),
              "base_model_name_or_path": job["repo"], "r": rank, "lora_alpha": scale * rank, "lora_dropout": 0.0, "bias": "none",
              "fan_in_fan_out": False, "inference_mode": True, "use_rslora": False, "target_modules": pattern}
    with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    if job.get("mlx_ok"):
        where = _mlx_layers_path(job)
        if where:
            _export_mlx(adapter, os.path.join(out_dir, "mlx"), job, where)
    return out_dir


def _mlx_layers_path(job: dict) -> str | None:
    """Where mlx-lm keeps this model's blocks (model.layers, or language_model.model.layers in a vision-language family), so the
    MLX copy of the adapter loads with mlx_lm.generate. Asked of mlx-lm when it is installed here; otherwise only a plain text
    model's path is known."""
    repo = job.get("mlx_repo") or job["repo"]
    try:
        from .backends.mlx_backend import layers_path
        return layers_path(repo, job["layers"])
    except ImportError:
        from .job import load_config
        return None if "text_config" in load_config(repo) else "model.layers"
    except Exception:
        return None


def _export_mlx(adapter: dict[str, np.ndarray], out_dir: str, job: dict, where: str = "model.layers"):
    """The same adapter in mlx-lm's format: `mlx_lm.generate --model <base> --adapter-path <out>/mlx`."""
    from safetensors.numpy import save_file
    os.makedirs(out_dir, exist_ok=True)
    tensors, modules = {}, set()
    for key, value in adapter.items():
        i, module, which = _split(key)
        modules.add(module)
        tensors[f"{where}.{i}.{module}.lora_{which.lower()}"] = np.ascontiguousarray(np.asarray(value, dtype=np.float32).T)
    save_file(tensors, os.path.join(out_dir, "adapters.safetensors"))
    config = {"fine_tune_type": "lora", "num_layers": job["layers"],
              "lora_parameters": {"rank": job["rank"], "scale": job["scale"], "dropout": 0.0, "keys": sorted(modules)}}
    with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def add_saved_modules(model, out_dir: str, modules: list[str]):
    """The modules a LoRA run trained in full, added to its PEFT adapter as modules_to_save."""
    from safetensors.numpy import load_file, save_file
    path = os.path.join(out_dir, "adapter_model.safetensors")
    tensors = load_file(path)
    for name, p in model.named_parameters():
        if any(name.startswith(m + ".") for m in modules):
            tensors[f"base_model.model.{name}"] = np.ascontiguousarray(p.detach().float().cpu().numpy())
    save_file(tensors, path)
    with open(os.path.join(out_dir, "adapter_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    config["modules_to_save"] = sorted(modules)
    with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


BLOCKS_FILE = "model-blocks.safetensors"
ENDS_FILE = "model-ends.safetensors"


def _dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def export_blocks(weights: dict[str, np.ndarray], out_dir: str, job: dict, dtype: str = "bfloat16") -> str:
    """The trained blocks of a full fine-tuning run, under their names in the model. The rest of the model is written by the
    training process (write_ends), which holds it."""
    import torch
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    stack = job["layout"]["stack"]
    target = _dtype(dtype)
    tensors = {}
    for key, value in weights.items():
        block, name = key.split(".", 1)
        tensors[f"{stack}.{block}.{name}"] = torch.tensor(np.ascontiguousarray(value)).to(target)
    save_file(tensors, os.path.join(out_dir, BLOCKS_FILE), metadata={"format": "pt"})
    return out_dir


def write_ends(model, out_dir: str, job: dict, dtype: str = "bfloat16"):
    """The modules around the blocks, the config, the index of both files and the tokenizer: the folder then loads with
    transformers like any model."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    stack = job["layout"]["stack"] + "."
    target = _dtype(dtype)
    seen, unique = set(), {}
    for k, v in model.state_dict().items():
        if k.startswith(stack) or k == "layerpool_anchor" or v.data_ptr() in seen:
            continue
        seen.add(v.data_ptr())
        unique[k] = v.detach().to(target).contiguous()
    save_file(unique, os.path.join(out_dir, ENDS_FILE), metadata={"format": "pt"})
    index = {}
    for name in (BLOCKS_FILE, ENDS_FILE):
        with safe_open(os.path.join(out_dir, name), framework="pt") as f:
            index.update({k: name for k in f.keys()})
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": index}, f, indent=1)
    config = load_config(job["repo"])
    for key, value in job.get("config", {}).items():
        setattr(config, key, value)
    if hasattr(config, "dtype") and "torch_dtype" not in config.__dict__:
        config.dtype = dtype
    else:
        config.torch_dtype = dtype
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.save_pretrained(out_dir)
    write_extras(job["repo"], out_dir)


def write_extras(repo: str, out_dir: str):
    """What the source model has besides its weights: the generation settings and the tokenizer, and the processor of an image or
    speech model (its image processor or feature extractor), so AutoProcessor and generate() work on the folder."""
    import transformers
    for kind in ("GenerationConfig", "AutoProcessor", "AutoTokenizer"):
        try:
            getattr(transformers, kind).from_pretrained(repo).save_pretrained(out_dir)
        except Exception:
            pass

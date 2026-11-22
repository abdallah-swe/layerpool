import zlib

import numpy as np

ONE_PLUS_NORM = ("gemma", "qwen3_next", "qwen3_5")


def random_weights(shapes: dict[str, tuple], config: dict) -> dict[str, np.ndarray]:
    """Initial values for training from scratch, from each weight's full-model name only: the same on every device and backend."""
    std = config.get("initializer_range") or 0.02
    out = {}
    one_plus = str(config.get("model_type") or "").startswith(ONE_PLUS_NORM)
    for name, shape in shapes.items():
        if len(shape) == 1:
            parts = name.lower().split(".")
            is_norm = "norm" in name.lower() or any(p == "ln" or p.startswith("ln_") or p.endswith("_ln") for p in parts)
            ones = is_norm and name.endswith("weight") and not one_plus
            out[name] = (np.ones if ones else np.zeros)(shape, np.float32)
        else:
            out[name] = np.random.default_rng(zlib.crc32(name.encode())).normal(0, std, shape).astype(np.float32)
    return out

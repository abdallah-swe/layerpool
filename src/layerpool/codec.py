import io
import json
import os
import time

import numpy as np


def encode(x: np.ndarray, mode: str = "int8", outliers: int = 8) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if mode == "fp32" or x.ndim == 0 or x.size == 0:
        return {"x": x}
    if mode == "fp16":
        if not np.isfinite(x).all():
            return {"x": x}
        scale = np.abs(x).max(axis=-1, keepdims=True).astype(np.float32) + 1e-30
        return {"h": (x / scale).astype(np.float16), "s": scale}
    if mode != "int8":
        raise ValueError(f"unknown mode {mode}")
    out = {}
    rest = x
    if outliers > 0 and x.shape[-1] > outliers:
        top = np.argsort(np.abs(x).reshape(-1, x.shape[-1]).max(axis=0))[-outliers:].astype(np.int32)
        rest = x.copy()
        rest[..., top] = 0
        out["oi"] = top
        out["ov"] = x[..., top]
    scale = np.abs(rest).max(axis=-1, keepdims=True).astype(np.float32) + 1e-30
    out["q"] = np.round(rest / scale * 127).astype(np.int8)
    out["s"] = scale
    return out


def decode(parts: dict[str, np.ndarray]) -> np.ndarray:
    if "x" in parts:
        return parts["x"].astype(np.float32)
    if "h" in parts:
        return parts["h"].astype(np.float32) * parts["s"]
    x = parts["q"].astype(np.float32) * parts["s"] / 127.0
    if "oi" in parts:
        x[..., parts["oi"]] = parts["ov"].astype(np.float32)
    return x


def pack(meta: dict | None = None, **tensors) -> bytes:
    buf = io.BytesIO()
    arrays = {k: np.asarray(v) for k, v in tensors.items()}
    if meta:
        arrays["__meta__"] = np.frombuffer(json.dumps(meta, default=float).encode(), dtype=np.uint8)
    np.savez(buf, **arrays)
    return buf.getvalue()


def unpack(data: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        items = {k: z[k] for k in z.files}
    meta = json.loads(items.pop("__meta__").tobytes().decode()) if "__meta__" in items else {}
    return meta, items


def split_prefix(arrays: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}


def with_prefix(arrays: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {prefix + k: v for k, v in arrays.items()}


def to_half(arrays: dict, dtype: str) -> tuple[dict, dict]:
    """float32 arrays as 16-bit values for sending: float16, or bfloat16 kept as its bits in uint16 (rounded to nearest even)."""
    out, kinds = {}, {}
    for k, v in arrays.items():
        v = np.asarray(v)
        if v.dtype != np.float32:
            out[k] = v
            continue
        if dtype == "float16":
            out[k], kinds[k] = v.astype(np.float16), "f16"
        else:
            bits = np.ascontiguousarray(v).view(np.uint32).astype(np.uint64)
            rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            out[k], kinds[k] = np.where(np.isnan(v), np.uint16(0x7FC0), rounded), "bf16"
    return {"half": kinds}, out


def from_half(meta: dict | None, arrays: dict) -> dict:
    """The float32 arrays back from to_half."""
    kinds = (meta or {}).get("half") or {}
    out = dict(arrays)
    for k, kind in kinds.items():
        v = arrays[k]
        out[k] = v.astype(np.float32) if kind == "f16" else (v.astype(np.uint32) << 16).view(np.float32)
    return out


def replace_file(src: str, dst: str, tries: int = 20):
    """os.replace, retried for a moment: on Windows an antivirus scan or another process reading the file can block it briefly."""
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(0.1 * (attempt + 1))


class DiskStore:
    """A dict of byte blobs kept in files in its own folder instead of in memory: checkpoints of a device's layers (and of a
    neighbour's) can be gigabytes in full fine-tuning."""

    def __init__(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.keys_: dict = {}

    def _path(self, key) -> str:
        return os.path.join(self.folder, "_".join(str(k) for k in key) + ".bin")

    def __setitem__(self, key, data: bytes):
        path = self._path(key)
        with open(path + ".part", "wb") as f:
            f.write(data)
        replace_file(path + ".part", path)
        self.keys_[key] = len(data)

    def __getitem__(self, key) -> bytes:
        if key not in self.keys_:
            raise KeyError(key)
        with open(self._path(key), "rb") as f:
            return f.read()

    def get(self, key, default=None):
        return self[key] if key in self.keys_ else default

    def __contains__(self, key) -> bool:
        return key in self.keys_

    def __delitem__(self, key):
        self.keys_.pop(key)
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass

    def __iter__(self):
        return iter(list(self.keys_))

    def __len__(self) -> int:
        return len(self.keys_)

    def clear(self):
        for key in list(self.keys_):
            del self[key]

    def close(self):
        import shutil
        self.keys_.clear()
        shutil.rmtree(self.folder, ignore_errors=True)


class Spilled:
    """Byte blobs written once to a file and read back on demand (the inputs and loss gradients of a step the pool may have to
    replay), so the replay log does not hold them in memory."""

    def __init__(self, path: str, blobs):
        self.path, self.sizes = path, []
        with open(path, "wb") as f:
            for blob in blobs:
                f.write(blob)
                self.sizes.append(len(blob))

    def __len__(self) -> int:
        return len(self.sizes)

    def __getitem__(self, i: int) -> bytes:
        i = range(len(self.sizes))[i]
        with open(self.path, "rb") as f:
            f.seek(sum(self.sizes[:i]))
            return f.read(self.sizes[i])

    def __iter__(self):
        with open(self.path, "rb") as f:
            for size in self.sizes:
                yield f.read(size)

    @property
    def nbytes(self) -> int:
        return sum(self.sizes)

    def drop(self):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass

"""Get only the tensors a device needs from a Hugging Face model.

A safetensors file starts with a JSON header that lists every tensor's byte range, so a device that holds 2 of 64 layers can
download the header and those byte ranges instead of the whole file. Files that are already on disk are read in place."""
import glob
import hashlib
import json
import os
import struct
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .codec import replace_file

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "layerpool", "tensors")
MERGE_GAP = 1 << 20
WHOLE_FILE_SHARE = 0.7


def _hub_file(repo: str, name: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name)


def load_json(repo: str, name: str) -> dict:
    path = os.path.join(repo, name) if os.path.isdir(repo) else _hub_file(repo, name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def weight_map(repo: str) -> dict[str, str]:
    """tensor name -> file name inside the repo."""
    try:
        return load_json(repo, "model.safetensors.index.json")["weight_map"]
    except Exception:
        pass
    try:
        return {key: "model.safetensors" for key in header(repo, "model.safetensors") if key != "__metadata__"}
    except Exception:
        if os.path.isdir(repo):
            files = os.listdir(repo)
        else:
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo)
        if not any(f.endswith(".safetensors") for f in files):
            other = sorted({os.path.splitext(f)[1] for f in files if f.endswith((".bin", ".pt", ".pth", ".gguf", ".ckpt"))})
            found = f" (only {', '.join(other)} files)" if other else ""
            raise ValueError(f"{repo} has no .safetensors weights{found}; layerpool "
                             "reads safetensors only. Pick a repo that has them, or load it once with transformers and save_pretrained "
                             "it to a folder (that writes safetensors)") from None
        raise


def _local(repo: str, name: str) -> str | None:
    if os.path.isdir(repo):
        return os.path.join(repo, name)
    try:
        from huggingface_hub import try_to_load_from_cache
        found = try_to_load_from_cache(repo, name)
        return found if isinstance(found, str) and os.path.exists(found) else None
    except Exception:
        return None


def _remote(repo: str, name: str) -> tuple[str, dict, str, int]:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    meta = get_hf_file_metadata(hf_hub_url(repo, name))
    url = meta.location or hf_hub_url(repo, name)
    hub = urllib.parse.urlsplit(hf_hub_url(repo, name)).hostname
    headers = build_hf_headers() if urllib.parse.urlsplit(url).hostname == hub else {}
    return url, headers, meta.commit_hash or "main", int(meta.size or 0)


def revision(repo: str) -> str:
    """The commit of a hub model (or the mtime of a local folder), to key caches by."""
    if os.path.isdir(repo):
        files = sorted(f for f in os.listdir(repo) if f.endswith((".safetensors", ".json")))
        return hashlib.sha1("|".join(f"{f}:{os.path.getsize(os.path.join(repo, f))}:{os.path.getmtime(os.path.join(repo, f))}"
                                     for f in files).encode()).hexdigest()[:12]
    return _remote(repo, "config.json")[2]


def _cached_parts(repo: str, name: str) -> list[str]:
    return sorted(glob.glob(os.path.join(CACHE, repo.replace("/", "--"), "*", f"{name.replace('/', '--')}.*.safetensors")),
                  key=os.path.getmtime, reverse=True)


def _get_range(url: str, headers: dict, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={**headers, "Range": f"bytes={start}-{end - 1}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as r:
                data = r.read()
            if r.status != 206 and start != 0:
                raise IOError(f"server ignored the byte range (HTTP {r.status})")
            data = data[:end - start]
            if len(data) != end - start:
                raise IOError(f"got {len(data)} bytes instead of {end - start}")
            return data
        except Exception:
            if attempt == 4:
                raise
    raise IOError("unreachable")


def _read_header(read) -> tuple[dict, int]:
    size = struct.unpack("<Q", read(0, 8))[0]
    return json.loads(read(8, 8 + size)), 8 + size


def header(repo: str, name: str) -> dict:
    local = _local(repo, name)
    if local:
        with open(local, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(size))
    try:
        url, headers, _, _ = _remote(repo, name)
    except Exception:
        cached = _cached_parts(repo, name)
        if not cached:
            raise
        with open(cached[0], "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(size))
    return _read_header(lambda a, b: _get_range(url, headers, a, b))[0]


CHUNK = 1 << 20


def _stream_range(url: str, headers: dict, start: int, end: int, sink):
    """Bytes start..end of a remote file, handed to sink(position, bytes) a megabyte at a time, so no whole range sits in memory.
    A failed attempt starts the range again; sink writes to fixed places, so that is safe."""
    request = urllib.request.Request(url, headers={**headers, "Range": f"bytes={start}-{end - 1}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as r:
                if r.status != 206 and start != 0:
                    raise IOError(f"server ignored the byte range (HTTP {r.status})")
                pos = start
                while pos < end:
                    chunk = r.read(min(CHUNK, end - pos))
                    if not chunk:
                        raise IOError(f"got {pos - start} bytes instead of {end - start}")
                    sink(pos, chunk)
                    pos += len(chunk)
            return
        except Exception:
            if attempt == 4:
                raise


def _layout(entries: list[tuple[str, dict, int]]) -> tuple[bytes, dict]:
    """The header of a safetensors file holding these tensors in this order, and where each tensor's bytes start after it."""
    out, offset, starts = {}, 0, {}
    for key, info, size in entries:
        out[key] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + size]}
        starts[key] = offset
        offset += size
    head = json.dumps({"__metadata__": {"format": "pt"}, **out}, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    return struct.pack("<Q", len(head)) + head, {k: len(head) + 8 + v for k, v in starts.items()}


def copy_tensors(files: dict[str, list[str]], rename, path: str):
    """A safetensors file at `path` with the tensors `files` names ({source file: [keys]}), renamed, copied as raw bytes a
    megabyte at a time instead of loaded."""
    entries, sources = [], []
    for source, keys in files.items():
        with open(source, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(size))
        for key in keys:
            a, b = head[key]["data_offsets"]
            entries.append((rename(key), head[key], b - a))
            sources.append((source, 8 + size + a, b - a))
    header, starts = _layout(entries)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.part"
    with open(tmp, "wb") as out:
        out.write(header)
        for (key, _, _), (source, at, size) in zip(entries, sources):
            with open(source, "rb") as f:
                f.seek(at)
                left = size
                while left:
                    chunk = f.read(min(CHUNK, left))
                    if not chunk:
                        raise IOError(f"{source} ends inside {key}")
                    out.write(chunk)
                    left -= len(chunk)
    replace_file(tmp, path)


def fetch(repo: str, keys, log=None) -> dict[str, list[str]]:
    """Make the tensors `keys` available on disk. Returns {safetensors path: [keys in that file]}."""
    log = log or (lambda *a: None)
    files = weight_map(repo)
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(files[key], []).append(key)
    out = {}
    for name, wanted in sorted(by_file.items()):
        local = _local(repo, name)
        if local:
            out[local] = wanted
            continue
        try:
            out[_fetch_part(repo, name, wanted, log)] = wanted
        except Exception as e:
            log(f"partial download of {name} failed ({e!r}); downloading the whole file")
            out[_hub_file(repo, name)] = wanted
    return out


def _already_here(folder: str, name: str, head: dict, wanted: list[str]) -> dict:
    """Tensors of this file that an earlier download of other layers (before a rebalance) already put on this device:
    {key: (file, offset, size)}, so only the rest is downloaded again."""
    found = {}
    for part in glob.glob(os.path.join(folder, f"{name.replace('/', '--')}.*.safetensors")):
        try:
            with open(part, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                theirs = json.loads(f.read(size))
        except (OSError, ValueError, struct.error):
            continue
        for key in wanted:
            info = theirs.get(key)
            if key in found or not isinstance(info, dict):
                continue
            if info.get("dtype") == head[key]["dtype"] and info.get("shape") == head[key]["shape"]:
                a, b = info["data_offsets"]
                found[key] = (part, 8 + size + a, b - a)
    return found


def _fetch_part(repo: str, name: str, wanted: list[str], log) -> str:
    key_hash = hashlib.sha1("\n".join(sorted(wanted)).encode()).hexdigest()[:16]
    try:
        url, headers, commit, size = _remote(repo, name)
    except Exception:
        for found in _cached_parts(repo, name):
            if found.endswith(f".{key_hash}.safetensors"):
                log(f"the hub cannot be reached; using the copy of {name} already on this device")
                return found
        raise
    folder = os.path.join(CACHE, repo.replace("/", "--"), commit)
    path = os.path.join(folder, f"{name.replace('/', '--')}.{key_hash}.safetensors")
    if os.path.exists(path):
        os.utime(path)
        return path
    head, base = _read_header(lambda a, b: _get_range(url, headers, a, b))
    everything = sorted((base + head[k]["data_offsets"][0], base + head[k]["data_offsets"][1], k) for k in wanted)
    here = _already_here(folder, name, head, wanted)
    spans = [s for s in everything if s[2] not in here]
    need = sum(b - a for a, b, _ in spans)
    if size and need > WHOLE_FILE_SHARE * size:
        return _hub_file(repo, name)
    if spans:
        kept = f", {len(here)} tensors already on this device" if here else ""
        log(f"downloading {need / 1e6:.0f} MB of {name} ({size / 1e6:.0f} MB), only the layers this device holds{kept}")
    groups, current = [], None
    for a, b, k in spans:
        if current and a - current[1] <= MERGE_GAP:
            current[1] = max(current[1], b)
            current[2].append((a, b, k))
        else:
            current = [a, b, [(a, b, k)]]
            groups.append(current)
    header, starts = _layout([(k, head[k], b - a) for a, b, k in everything])
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.part"
    with open(tmp, "wb") as f:
        f.write(header)
        f.truncate(len(header) + sum(b - a for a, b, _ in everything))
        for key, (source, at, length) in here.items():
            f.seek(starts[key])
            with open(source, "rb") as src:
                src.seek(at)
                left = length
                while left:
                    chunk = src.read(min(CHUNK, left))
                    if not chunk:
                        raise IOError(f"{source} ends inside {key}")
                    f.write(chunk)
                    left -= len(chunk)

    def one(group):
        _, _, members = group
        with open(tmp, "r+b") as f:
            def sink(pos, chunk):
                for a, b, k in members:
                    lo, hi = max(a, pos), min(b, pos + len(chunk))
                    if lo < hi:
                        f.seek(starts[k] + lo - a)
                        f.write(chunk[lo - pos:hi - pos])
            _stream_range(url, headers, group[0], group[1], sink)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(one, groups))
    except BaseException:
        os.remove(tmp)
        raise
    replace_file(tmp, path)
    _prune_parts(folder, name, path)
    return path


def _prune_parts(folder: str, name: str, keep: str):
    """Partial downloads of one file pile up as a device's layers change. The newest two stay (the next change usually reuses
    the previous range); older ones go once unused for ten minutes, since another device on this machine may be reading one."""
    parts = [p for p in glob.glob(os.path.join(folder, f"{name.replace('/', '--')}.*.safetensors")) if p != keep]
    parts.sort(key=os.path.getmtime, reverse=True)
    now = time.time()
    for part in parts[1:]:
        if now - os.path.getmtime(part) > 600:
            try:
                os.remove(part)
            except OSError:
                pass

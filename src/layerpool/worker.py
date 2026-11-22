import argparse
import faulthandler
import gc
import hashlib
import json
import os
import platform
import queue
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import traceback
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import __version__
from .blocks import BlockCore, load_blocks, mlx_masks, unpack_args, unwrap_lora
from .codec import DiskStore, decode, encode, pack, split_prefix, to_half, unpack, with_prefix
from .net import Cancelled, HTTPError, Peer, PeerServer, Sender
from .telemetry import sample
from .tunnel import Tunnel, child_options

ROCM_BF16 = ("gfx90a", "gfx94", "gfx95", "gfx11", "gfx12")


def native_bf16() -> bool:
    """bfloat16 in hardware: NVIDIA Ampere (compute capability 8) and newer, AMD MI200/MI300 and RDNA3 and newer.
    torch.cuda.is_bf16_supported() also says yes on cards that only emulate it (a T4, older AMD cards), which is slow."""
    import torch
    if not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None):
        return all(str(getattr(torch.cuda.get_device_properties(i), "gcnArchName", "")).startswith(ROCM_BF16)
                   for i in range(torch.cuda.device_count()))
    return all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count()))


def train_dtype(device: str):
    """The 16-bit type a device trains LoRA in: bfloat16 where it is native, float16 where float16 is fast (with loss scaling),
    and float32 on GPUs where neither is (GTX 10xx and older) and on CPUs."""
    import torch
    if device == "mps":
        newer = getattr(torch.backends.mps, "is_macos_or_newer", None)
        return torch.bfloat16 if newer is None or newer(14, 0) else torch.float16
    if device == "cpu":
        return torch.float32
    if native_bf16():
        return torch.bfloat16
    if not getattr(torch.version, "hip", None) and min(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())) < (7, 0) \
            and min(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())) != (6, 0):
        return torch.float32
    return torch.float16


def _arch(name: str) -> tuple | None:
    """("sm_86" or "compute_90a") -> (8, 6) or (9, 0); GPUs run code built for an earlier minor version of their major version."""
    digits = re.match(r"(?:sm|compute)_(\d+)(\d)", name)
    return (int(digits.group(1)), int(digits.group(2))) if digits else None


_HIP_RUNS: dict = {}


def cuda_problem() -> str | None:
    """Why this machine's NVIDIA/AMD GPU cannot train with the installed PyTorch, or None when it can (or there is none)."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        if shutil.which("nvidia-smi") and platform.system() in ("Windows", "Linux"):
            return ("this machine has an NVIDIA GPU, but the installed PyTorch has no CUDA support, so it trains on the CPU. Install "
                    "the CUDA build: https://pytorch.org/get-started/locally/")
        if platform.system() == "Linux" and (os.path.exists("/dev/kfd") or shutil.which("rocminfo") or shutil.which("amd-smi")):
            return ("this machine has an AMD GPU, but the installed PyTorch has no ROCm support, so it trains on the CPU. Install the "
                    "ROCm build (choose ROCm at https://pytorch.org/get-started/locally/), then layerpool")
        return None
    if getattr(torch.version, "hip", None):
        for i in range(torch.cuda.device_count()):
            if _HIP_RUNS.get(i) is True:
                continue
            try:
                probe = torch.ones(8, device=f"cuda:{i}") * 2
                torch.cuda.synchronize(i)
                assert float(probe.sum().cpu()) == 16.0
                _HIP_RUNS[i] = True
            except Exception as e:
                arch = str(getattr(torch.cuda.get_device_properties(i), "gcnArchName", "")).split(":")[0]
                return (f"{torch.cuda.get_device_name(i)} ({arch}) does not run the installed PyTorch {torch.__version__} ({e.__class__.__name__}), "
                        "so this device trains on the CPU. Install a ROCm build of PyTorch made for it, or, for a card ROCm does not list, "
                        "set HSA_OVERRIDE_GFX_VERSION (for example 10.3.0 for an RX 6000 card) before starting layerpool")
        return None
    built = torch.cuda.get_arch_list()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        native = any(_arch(a) and _arch(a)[0] == major and _arch(a)[1] <= minor for a in built if a.startswith("sm_"))
        ptx = any(_arch(a) and _arch(a) <= (major, minor) for a in built if a.startswith("compute_"))
        if not native and not ptx:
            return (f"{torch.cuda.get_device_name(i)} (compute capability {major}.{minor}) is not supported by PyTorch {torch.__version__}, "
                    "so this device trains on the CPU. Install a PyTorch build for it, for example: "
                    "pip install \"torch<2.15\" --index-url https://download.pytorch.org/whl/cu126")
    return None


def out_of_memory(error: BaseException) -> bool:
    """Whether an error means the device ran out of memory (CUDA, Apple GPU with PyTorch or MLX, or the CPU)."""
    if type(error).__name__ in ("OutOfMemoryError", "MemoryError"):
        return True
    text = str(error).lower()
    return "out of memory" in text or "outofmemory" in text or "[metal::malloc]" in text or "insufficient memory" in text


def library_versions() -> dict:
    """The versions that decide how a model is built and run here."""
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "mlx", "mlx-lm", "bitsandbytes", "flash-linear-attention", "causal-conv1d"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    try:
        from transformers.utils.import_utils import BITSANDBYTES_MIN_VERSION
        out["bitsandbytes-needed"] = BITSANDBYTES_MIN_VERSION
    except ImportError:
        pass
    return out


STORE_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "layerpool", "store")


def _store_folder() -> str:
    """This device's own folder for the checkpoints it keeps; folders of devices that are no longer running are removed."""
    import psutil
    os.makedirs(STORE_ROOT, exist_ok=True)
    for name in os.listdir(STORE_ROOT):
        pid = name.split("_", 1)[0]
        if pid.isdigit() and not psutil.pid_exists(int(pid)):
            shutil.rmtree(os.path.join(STORE_ROOT, name), ignore_errors=True)
    return os.path.join(STORE_ROOT, f"{os.getpid()}_{secrets.token_hex(4)}")


def unit_of(key: str) -> int:
    return int(key.split(".", 1)[0])


class Abort(Exception):
    pass


class NeighbourDown(Exception):
    def __init__(self, wid: str, error: Exception):
        super().__init__(f"{wid} unreachable: {error!r}")
        self.wid = wid


def _has_mlx() -> bool:
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except ImportError:
        return False


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and cuda_problem() is None
    except ImportError:
        return False


def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


def pick_backend(backend: str = "auto", cpu: bool = False) -> str:
    if backend != "auto":
        return backend
    if not cpu and _has_cuda():
        return "torch"
    return "mlx" if _has_mlx() else "torch"


def total_ram_gb() -> float:
    """The machine's memory, or a container's limit when it runs in one (Docker, Kubernetes, some notebook services)."""
    import psutil
    total = psutil.virtual_memory().total
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path, encoding="utf-8") as f:
                value = f.read().strip()
        except OSError:
            continue
        if value.isdigit() and 0 < int(value) < total:
            total = int(value)
    return round(total / 1e9, 1)


def cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        if platform.system() == "Windows":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def default_name() -> str:
    """The host name, or the notebook service a device runs in (their host names are random)."""
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None:
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    return socket.gethostname()


def keep_awake():
    """Keep this computer from sleeping while it lends itself to a pool (a sleeping device looks gone and costs a rollback).
    Closing a laptop's lid still sleeps it. Returns the helper process to stop, if any."""
    try:
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            return subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        elif shutil.which("systemd-inhibit") and os.environ.get("XDG_SESSION_ID"):
            return subprocess.Popen(["systemd-inhibit", "--what=sleep", "--who=layerpool", "--why=training", "--mode=block",
                                     "sleep", "infinity"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **child_options())
    except Exception:
        pass
    return None


def amd_4bit_works(index: int) -> bool:
    """Whether bitsandbytes' 4-bit layers run on this AMD GPU (its recent versions support many of them)."""
    try:
        import torch
        import bitsandbytes.functional as bnb
        packed, state = bnb.quantize_4bit(torch.randn(64, 64, device=f"cuda:{index}", dtype=torch.float16), quant_type="nf4")
        back = bnb.dequantize_4bit(packed, state)
        torch.cuda.synchronize(index)
        return bool(torch.isfinite(back).all())
    except Exception:
        return False


def describe_devices(backend: str, cpu: bool = False) -> list[dict]:
    ram = total_ram_gb()
    if backend == "torch" and not cpu and _has_cuda():
        import torch
        amd = bool(getattr(torch.version, "hip", None))
        out = []
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            out.append({"kind": "cuda", "name": torch.cuda.get_device_name(i), "gb": round(total / 1e9, 1), "amd": amd,
                        **({"q4": amd_4bit_works(i)} if amd else {})})
        return out
    if backend == "mlx" and not cpu:
        return [{"kind": "mlx", "name": cpu_name() or platform.machine(), "gb": ram}]
    if backend == "torch" and not cpu and _has_mps():
        return [{"kind": "mps", "name": cpu_name() or platform.machine(), "gb": ram}]
    return [{"kind": "cpu", "name": cpu_name(), "gb": ram}]


def model_key(job: dict, first: int, last: int) -> tuple:
    return (job["repo"], job.get("mlx_repo"), job.get("auto"), job["kind"], job.get("mode", "lora"), job.get("random_init", False), first, last)


class Worker:
    def __init__(self, coordinator: str, token: str, name: str | None = None, cap_gb: float | None = None, tunnel: bool | None = False,
                 port: int = 0, backend: str = "auto", cpu: bool = False, identity: str | None = None, log=None, advertise: list | None = None,
                 layers: str | None = None, gpus: str | None = None, gpu_caps: str | None = None, order: int | None = None,
                 allow_sleep: bool = False, ram_cap: float | None = None):
        self.server = PeerServer(port=port)
        self.coord = Peer([coordinator], token)
        self.name = name or default_name()
        self.cap_gb = cap_gb
        self.allow_sleep = allow_sleep
        self.asked = {k: v for k, v in (("layers", layers), ("gpus", gpus), ("gpu_caps", gpu_caps), ("order", order), ("ram_cap", ram_cap))
                      if v is not None}
        self.backend, self.cpu = pick_backend(backend, cpu), cpu
        self.devices = describe_devices(self.backend, cpu)
        self.identity = identity
        problem = None if cpu else cuda_problem()
        if problem:
            (log or (lambda *a: print("[layerpool]", *a, flush=True)))(problem)
        self.addrs = [self.server.url_at(a) for a in advertise] if advertise else [self.server.lan_url()]
        self.tunnel = None
        self.tunnel_wanted = tunnel
        self.tunnel_log = log or (lambda *a: print("[layerpool]", *a, flush=True))
        if tunnel:
            self._open_tunnel()
        self.alive = True
        self.chain = None
        self.loaded = None
        self.key = None
        self.run_id = None
        self.state_step = None
        self.base_seen = None
        self.synced_epoch = None
        self.store = DiskStore(_store_folder())
        self.store_lock = threading.Lock()
        self.keep_from = -1
        self.replica = None
        self.settings: dict = {}
        self.server.routes.update({("GET", "/ckpt"): self._serve_ckpt, ("POST", "/replica"): self._take_replica})
        self.log = log or (lambda *a: print("[layerpool]", *a, flush=True))

    def _ask_pool(self, path: str, patience: float | None = None) -> dict:
        """Ask the pool, riding out short outages of the network or of a tunnel instead of leaving at once."""
        patience = patience or float(os.environ.get("LAYERPOOL_PATIENCE_S", "1800"))
        start, warned = time.time(), False
        while True:
            try:
                out = self.coord.json(path)
                if warned:
                    self.log("the pool answers again")
                return out
            except ConnectionError:
                if time.time() - start > patience:
                    raise
                if not warned:
                    self.log(f"the pool does not answer; trying again for up to {patience / 60:.0f} minutes")
                    warned = True
                time.sleep(5)

    def _open_tunnel(self):
        self.tunnel = Tunnel(self.server.port, log=self.tunnel_log)
        self.addrs.append(self.tunnel.url)

    def join(self):
        if self.tunnel_wanted is None and self.tunnel is None:
            try:
                if self.coord.json("/info").get("tunnel"):
                    self.tunnel_log("the pool also has devices on the internet, so this device opens a tunnel too")
                    self._open_tunnel()
            except HTTPError:
                pass
        if not self.coord.token:
            self._ask_to_join()
        me = self.coord.json("/join", {"name": self.name, "addrs": self.addrs, "token": self.server.token, "devices": self.devices, "cap_gb": self.cap_gb,
                                         "near": self.coord.urls[0].startswith("http://"), "backend": self.backend,
                                         "version": __version__, "identity": self.identity,
                                         "libraries": library_versions(), **self.asked})
        self.wid = me["wid"]
        if isinstance(me.get("settings"), dict):
            self._apply({"settings": me["settings"]})
        if me.get("token"):
            self.coord.token = me["token"]
        self.heartbeat_every = me.get("heartbeat", 5.0)
        self.log(f"joined as {self.wid} ({self.name}, {[d['name'] for d in self.devices]})")
        if not getattr(self, "_beating", False):
            self._beating = True
            threading.Thread(target=self._heartbeat, daemon=True).start()

    def _ask_to_join(self):
        """No join code: ask the pool, and wait until someone there approves this device."""
        req = self.coord.json("/request", {"name": self.name, "devices": [d["name"] for d in self.devices]})
        self.log(f"asked to join. Approve this device on the pool's computer: in the dashboard, or with\n\n    layerpool approve {req['code']}\n")
        deadline = time.time() + req.get("expires_in_s", 900)
        while time.time() < deadline:
            got = self.coord.json(f"/request?ticket={req['ticket']}")
            if got["status"] == "approved":
                self.coord.token = got["invite"]
                return
            if got["status"] == "denied":
                raise HTTPError(self.coord.urls[0], 403, "the pool denied this device")
            time.sleep(2)
        raise HTTPError(self.coord.urls[0], 403, "nobody approved this device in time")

    def _telemetry(self) -> dict:
        return sample(self.backend, self.cpu or self.devices[0]["kind"] == "cpu")

    def _heartbeat(self):
        while self.alive:
            try:
                body = json.dumps({"telemetry": self._telemetry(), "applied": self.settings.get("version", 0)}).encode()
                reply = self.coord.request(f"/hb?w={self.wid}", body, timeout=20, retries=1)
                if reply:
                    self._apply(json.loads(reply))
            except Exception:
                pass
            time.sleep(self.heartbeat_every)

    def _apply(self, reply: dict):
        wanted = reply.get("settings")
        if not wanted or wanted.get("version", 0) == self.settings.get("version", 0):
            return
        self.settings = wanted
        if wanted.get("name"):
            self.name = wanted["name"]
        if "cap_gb" in wanted:
            self.cap_gb = wanted["cap_gb"]
        self._limit_memory()

    def _gpus(self) -> list[int]:
        """The GPUs this device trains on: the ones chosen in its settings, or all of them."""
        import torch
        count = torch.cuda.device_count()
        chosen = [i for i in (self.settings.get("gpus") or []) if 0 <= i < count]
        return chosen or list(range(count))

    def _limit_cuda(self):
        """Each GPU's share of the device's cap (or its own cap); GPUs left out and GPUs without a cap are not limited."""
        import torch
        used = self._gpus()
        caps = self.settings.get("gpu_caps") or {}
        totals = {i: torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())}
        for i, total in totals.items():
            gb = caps.get(str(i))
            if gb is None and self.cap_gb and i in used:
                gb = self.cap_gb / len(used)
            torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 1e9 / total) if gb else 1.0, i)

    def _limit_memory(self):
        """The memory cap, on whatever this device trains with right now: MLX, or PyTorch on CUDA or on the Mac GPU. Without a
        cap, a limit set earlier in this process (such limits are process-wide) is lifted."""
        if not self.cap_gb and not (self.settings.get("gpu_caps") and _has_cuda() and not self.cpu):
            try:
                if getattr(self, "engine", None) == "mlx":
                    import mlx.core as mx
                    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
                    if info.get("max_recommended_working_set_size"):
                        mx.set_memory_limit(int(1.5 * info["max_recommended_working_set_size"]))
                elif getattr(self, "engine", None) == "torch" and _has_cuda() and not self.cpu:
                    self._limit_cuda()
                elif getattr(self, "engine", None) == "torch" and not self.cpu and _has_mps():
                    import torch
                    torch.mps.set_per_process_memory_fraction(0.0)
            except Exception as e:
                self.log(f"could not lift the memory cap: {e!r}")
            return
        try:
            if getattr(self, "engine", None) == "mlx":
                import mlx.core as mx
                mx.set_memory_limit(int(self.cap_gb * 1e9))
            elif _has_cuda() and not self.cpu:
                self._limit_cuda()
            elif not self.cpu and getattr(self, "engine", None) == "torch":
                import torch
                if torch.backends.mps.is_available() and hasattr(torch.mps, "recommended_max_memory"):
                    torch.mps.set_per_process_memory_fraction(min(2.0, self.cap_gb * 1e9 / torch.mps.recommended_max_memory()))
        except Exception as e:
            self.log(f"could not apply the memory cap: {e!r}")

    def _release(self, keep: set | None = None):
        if self.chain is not None:
            if self.state_step is not None and self.state_step >= 0:
                packed = self._pack(self._snapshot())
                self._keep(self.state_step, packed)
                leaving = [u for u in packed if keep is None or u not in keep]
                if leaving:
                    try:
                        self.coord.request(f"/have?w={self.wid}&run={self.run_id}",
                                           json.dumps({"step": self.state_step, "layers": leaving, "holders": [self.wid]}).encode())
                    except Exception as e:
                        self.log(f"could not register the checkpoint of step {self.state_step}: {e!r}")
            self.chain = None
            self.loaded = None
            self.key = None
            self.state_step = None
            self.synced_epoch = None
        self._free_caches()

    def _free_caches(self):
        gc.collect()
        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        if "mlx.core" in sys.modules:
            import mlx.core as mx
            mx.clear_cache()

    def _peak_gb(self) -> float:
        """The most memory training used here since the last check, then the count starts again."""
        peak = 0.0
        try:
            if self.cpu or self.devices[0]["kind"] == "cpu":
                import psutil
                return psutil.Process().memory_info().rss / 1e9
            if "torch" in sys.modules:
                import torch
                if torch.cuda.is_available():
                    for i in self._gpus():
                        peak += torch.cuda.max_memory_allocated(i) / 1e9
                        torch.cuda.reset_peak_memory_stats(i)
                elif torch.backends.mps.is_available():
                    peak = max(peak, torch.mps.driver_allocated_memory() / 1e9)
            if "mlx.core" in sys.modules:
                import mlx.core as mx
                peak = max(peak, mx.get_peak_memory() / 1e9)
                mx.reset_peak_memory()
        except Exception:
            pass
        return peak

    def _units(self, first: int, last: int) -> set:
        return set(range(first, last))

    def _mlx(self, job: dict) -> bool:
        """MLX for text models whose blocks mlx-lm computes as transformers does (checked once per architecture on a small
        copy); PyTorch on the Mac GPU for the rest. A 4-bit run has no PyTorch way on a Mac, so then this Mac sits it out."""
        if self.backend != "mlx" or not job.get("mlx_ok"):
            return False
        key = (job["repo"], job.get("auto"), job.get("mode", "lora"))
        verdicts = self.__dict__.setdefault("_mlx_verdicts", {})
        if key not in verdicts:
            from .backends.mlx_backend import verify
            verdicts[key] = verify(job["repo"], job.get("auto", "AutoModelForCausalLM"), job.get("mode", "lora"), log=self.log)
            if verdicts[key]:
                self.log(f"MLX does not compute this model's blocks as transformers does ({verdicts[key]}), so this Mac trains them "
                         "with PyTorch")
        if verdicts[key] and job.get("kind") == "q4":
            raise RuntimeError(f"this Mac cannot train this 4-bit model: MLX does not compute its blocks as transformers does "
                               f"({verdicts[key]}), and PyTorch on a Mac has no 4-bit layers")
        return not verdicts[key]

    def _build_mlx(self, job: dict, first: int, last: int) -> list:
        import mlx.core as mx
        from .backends.mlx_backend import StageCore, load_range
        self.engine = "mlx"
        self._limit_memory()
        full = job.get("mode") == "full"
        loader = lambda: load_range(job.get("mlx_repo") or job["repo"], first, last, job["layout"]["n"], job["layout"]["stack"],
                                    mx.float32 if full else None,
                                    random_init=job.get("random_init", False), log=self.log)
        return [StageCore(loader, first, job["rank"], job["scale"], job["lr"], loss_scale=job.get("loss_scale", 1024.0),
                          checkpointing=job.get("checkpointing", True), device=mx.cpu if self.cpu else mx.gpu, mode=job.get("mode", "lora"),
                          weight_decay=job.get("weight_decay", 0.0), targets=job.get("target_modules"))]

    def _build_torch(self, job: dict, first: int, last: int, on_gpu: int | None = None) -> tuple[list, list]:
        import torch
        self.engine = "torch"
        self._limit_memory()
        mode = job.get("mode", "lora")
        loss_scale = job.get("loss_scale", 1024.0)
        if torch.cuda.is_available() and not self.cpu:
            devices = [f"cuda:{i}" for i in self._gpus()]
        elif not self.cpu and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            devices = ["mps"]
        else:
            devices = ["cpu"]
        spill_from = last if on_gpu is None or not devices[0].startswith("cuda") else first + on_gpu
        n = spill_from - first
        caps = self.settings.get("gpu_caps") or {}
        index = lambda d: int(d.split(":")[1])
        share = self.cap_gb * 1e9 / len(devices) if self.cap_gb else float("inf")
        room = [min(torch.cuda.mem_get_info(index(d))[0], caps[str(index(d))] * 1e9 if str(index(d)) in caps else share)
                for d in devices] if devices[0].startswith("cuda") else [1.0] * len(devices)
        share = [sum(room[:i]) / sum(room) for i in range(len(devices) + 1)]
        parts = [(first + round(share[i] * n), first + round(share[i + 1] * n), d) for i, d in enumerate(devices)]
        parts = [(a, b, d) for a, b, d in parts if b > a]
        if spill_from < last:
            parts.append((spill_from, last, "cpu"))
        cores = []
        kwargs = dict(loss_scale=loss_scale, checkpointing=job.get("checkpointing", True), mode=mode, weight_decay=job.get("weight_decay", 0.0),
                      targets=job.get("target_modules"))
        quantized = job["kind"] == "q4"
        half = train_dtype(devices[0].split(":")[0])
        dtype = torch.float32 if mode == "full" else half
        for a_, b_, d in parts:
            here = torch.float32 if d == "cpu" else (half if half != torch.float32 else torch.float16) if quantized else dtype
            blocks = load_blocks(job["repo"], job["layout"], a_, b_, d, here, quantized=quantized, random_init=job.get("random_init", False),
                                 log=self.log)
            cores.append(BlockCore(blocks, a_, job["rank"], job["scale"], job["lr"], **kwargs))
            cores[-1].spill = d == "cpu" and devices[0] != "cpu"
        return cores, [d for _, _, d in parts]

    def _build(self, a: dict):
        job, (first, last) = a["job"], a["range"]
        self._release(self._units(first, last))
        t = time.time()
        if self._mlx(job):
            cores, devices = self._build_mlx(job, first, last), ["mlx"]
        else:
            cores, devices = self._build_torch(job, first, last, a.get("on_gpu"))
        self.chain = cores
        self.loaded = (first, last, a.get("on_gpu"))
        self.key = model_key(job, first, last)
        self.built_gpus = (self.settings.get("gpus"), self.settings.get("gpu_caps"))
        self.state_step, self.base_seen = -1, None
        self.log(f"loaded layers {first}-{last} on {devices[:len(cores)]} in {time.time() - t:.0f}s")

    def _rewrap(self, job: dict):
        cores = []
        kwargs = dict(loss_scale=job.get("loss_scale", 1024.0), checkpointing=job.get("checkpointing", True), mode="lora", weight_decay=job.get("weight_decay", 0.0),
                      targets=job.get("target_modules"))
        for core in self.chain:
            if self._mlx(job):
                from .backends.mlx_backend import StageCore
                import mlx.core as mx
                held = (core.model, core.layers)
                cores.append(StageCore(lambda held=held: held, core.first, job["rank"], job["scale"], job["lr"],
                                       device=mx.cpu if self.cpu else mx.gpu, **kwargs))
            else:
                cores.append(BlockCore(unwrap_lora(core.layers), core.first, job["rank"], job["scale"], job["lr"], **kwargs))
                cores[-1].spill = getattr(core, "spill", False)
        self.chain = cores
        self.state_step, self.base_seen = -1, None
        self.log("kept the loaded model for the new run")

    def _serve_ckpt(self, query, body):
        with self.store_lock:
            data = self.store.get((int(query["layer"]), int(query["step"])))
        return (200, data) if data is not None else (404, b"")

    def _take_replica(self, query, body):
        if query.get("run") != self.run_id:
            return 409, {"error": "a checkpoint copy of another run"}
        with self.store_lock:
            self.store[(int(query["layer"]), int(query["step"]))] = body
        return 200, b""

    def _pack(self, flat: dict) -> dict[int, bytes]:
        return {u: pack(None, **{k: v for k, v in flat.items() if unit_of(k) == u}) for u in sorted({unit_of(k) for k in flat})}

    def _keep(self, step: int, packed: dict[int, bytes]):
        with self.store_lock:
            for u, data in packed.items():
                self.store[(u, step)] = data

    def _prune(self, keep_from: int):
        self.keep_from = max(self.keep_from, keep_from)
        with self.store_lock:
            own = sorted({st for _, st in self.store})[-2:]
            for key in [k for k in self.store if k[1] < self.keep_from and k[1] not in own]:
                del self.store[key]

    def _fetch(self, unit: int, step: int, epoch: int, patience: float = 900.0) -> bytes:
        with self.store_lock:
            if (unit, step) in self.store:
                return self.store[(unit, step)]
        deadline = time.time() + patience
        while True:
            try:
                where = self.coord.json(f"/where?w={self.wid}&layer={unit}&step={step}&epoch={epoch}&wait=60")
            except urllib.error.HTTPError as e:
                if e.code != 404 or time.time() > deadline:
                    raise
                continue
            for holder in where["holders"]:
                try:
                    if holder["w"] == "coord":
                        return self.coord.get_blob(f"/ckpt?layer={unit}&step={step}")
                    if holder["w"] != self.wid:
                        return Peer(holder["addrs"], holder["token"]).get_blob(f"/ckpt?layer={unit}&step={step}")
                except Exception as e:
                    self.log(f"could not fetch layer {unit} from {holder['w']}: {e!r}")
            if time.time() > deadline:
                raise ConnectionError(f"no holder of layer {unit} at step {step} could be reached")
            time.sleep(2)

    def _restore(self, step: int, epoch: int, units: set, base=None):
        t = time.time()
        state, fetched = {}, 0
        if step == -1 and not base:
            for core in self.chain:
                state.update(core.initial())
        else:
            for u in sorted(units):
                with self.store_lock:
                    local = None if step == -1 else self.store.get((u, step))
                if local is None:
                    local, fetched = self._fetch(u, step, epoch), fetched + 1
                    self._keep(step, {u: local})
                state.update(unpack(local)[1])
        for core in self.chain:
            core.zero_grad()
            restored = core.load_state(state)
            expected = len(core.params) if hasattr(core, "params") else restored
            if step >= 0 and restored < expected:
                raise RuntimeError(f"the checkpoint of layers {core.first}-{core.first + len(core.layers) - 1} has {restored} of the {expected} weights this "
                                   "device trains; the devices of this pool build the model differently (use the same transformers version)")
        self.state_step = step
        self.base_seen = base
        source = "the initial values" if step == -1 and not base else f"{len(units) - fetched} parts held here, {fetched} fetched"
        self.log(f"restored the state after step {step} ({source}) in {time.time() - t:.1f}s")

    def _zero_grad(self):
        for stage in self.chain or []:
            stage.zero_grad()

    def _snapshot(self) -> dict:
        flat = {}
        for core in self.chain or []:
            flat.update(core.state())
        return flat

    def _checkpoint(self, step: int, packed: dict[int, bytes], target: str, run_id: str | None = None):
        """Send a checkpoint to the pool or to the neighbour that keeps a copy, under the run it was taken in."""
        run_id = run_id or self.run_id
        holders = [self.wid]
        replica = None if target == "coord" else self.replica
        try:
            if replica is None:
                for u, data in packed.items():
                    self.coord.post_blob(f"/ckpt?w={self.wid}&run={run_id}&layer={u}&step={step}", data)
            else:
                peer = Peer(replica["addrs"], replica["token"])
                for u, data in packed.items():
                    peer.post_blob(f"/replica?run={run_id}&layer={u}&step={step}", data, retries=3)
                holders.append(replica["w"])
        except Exception as e:
            self.log(f"checkpoint copy of step {step} failed: {e!r}")
        try:
            self.coord.request(f"/have?w={self.wid}&run={run_id}", json.dumps({"step": step, "layers": list(packed), "holders": holders}).encode())
        except Exception as e:
            self.log(f"could not register checkpoint of step {step}: {e!r}")

    def run(self):
        self.join()
        awake = None if self.allow_sleep else keep_awake()
        told = None
        try:
            while self.alive:
                if self.tunnel is not None and not self.tunnel.alive():
                    self.log("this device's tunnel stopped; opening a new one and joining the pool again")
                    self.addrs = [u for u in self.addrs if u != self.tunnel.url]
                    self._release()
                    try:
                        self._open_tunnel()
                    except Exception as e:
                        self.log(f"could not open a new tunnel ({str(e).splitlines()[0]}); trying again in a minute")
                        time.sleep(60)
                        continue
                    self.join()
                    continue
                try:
                    a = self._ask_pool(f"/assignment?w={self.wid}")
                except HTTPError as e:
                    if e.code != 409:
                        raise
                    self.log("the pool restarted; joining it again")
                    self._release()
                    self.join()
                    continue
                if a.get("left_out") and a["left_out"] != told:
                    told = a["left_out"]
                    self.log(f"left out of this run after repeated errors ({told}); it will be used again in the next run")
                if a.get("done"):
                    break
                if a.get("wait"):
                    if self.chain is not None and a.get("run") is not None and a.get("run") == self.run_id:
                        try:
                            self._release()
                            self.log("left out of the current split; handed this device's layers to the others")
                        except Exception as e:
                            self.log("could not hand over the layers:", repr(e))
                            self.chain = None
                    time.sleep(0.5)
                    continue
                self._apply(a)
                self.replica = a.get("replica")
                self._prune(a.get("keep_from", -1))
                job, (first, last) = a["job"], a["range"]
                try:
                    if a["run"] != self.run_id:
                        with self.store_lock:
                            self.store.clear()
                        self.keep_from = -1
                        self.state_step = None
                        if self.chain is not None and self.key == model_key(job, first, last) and job.get("mode", "lora") == "lora":
                            self._rewrap(job)
                        else:
                            self._release()
                        self.run_id = a["run"]
                        self.synced_epoch = None
                    if self.loaded != (first, last, a.get("on_gpu")) or (self.chain is not None and getattr(self, "built_gpus", None) is not None
                                                        and self.built_gpus != (self.settings.get("gpus"), self.settings.get("gpu_caps"))):
                        self._build(a)
                    if self.synced_epoch != a["epoch"]:
                        if self.state_step != a["restore"] or (a["restore"] == -1 and self.base_seen != job.get("base")):
                            self._restore(a["restore"], a["epoch"], self._units(first, last), job.get("base"))
                        self.synced_epoch = a["epoch"]
                    self.coord.request(f"/ready?w={self.wid}&epoch={a['epoch']}", b"")
                    self._failures = 0
                    self._serve(a)
                except Abort:
                    self._zero_grad()
                except Exception as e:
                    self._zero_grad()
                    try:
                        now = self.coord.json(f"/epoch?w={self.wid}")
                    except Exception:
                        raise e
                    if now["epoch"] != a["epoch"] or now["finished"]:
                        self.log(f"plan {a['epoch']} ended while working ({e!r})")
                        continue
                    oom = out_of_memory(e)
                    peak = self._peak_gb() if oom else None
                    self._free_caches()
                    if oom:
                        self.log(f"ran out of memory (about {peak:.1f} GB in use): asking for fewer layers")
                    else:
                        self.log("error:", repr(e))
                        traceback.print_exc()
                    blame = f"&blame={e.wid}" if isinstance(e, NeighbourDown) else ""
                    extra = f"&oom={peak:.2f}" if oom else ""
                    if not oom and not blame and getattr(self, "engine", None) == "mlx" and job.get("kind") != "q4":
                        self.__dict__.setdefault("_mlx_verdicts", {})[(job["repo"], job.get("auto"), job.get("mode", "lora"))] = repr(e)
                        self.log(f"MLX failed on this model ({e!r}); this Mac goes on with PyTorch")
                        self._release()
                        extra = "&switched=1"
                    self.coord.request(f"/fail?w={self.wid}{blame}{extra}", repr(e).encode())
                    self._failures = getattr(self, "_failures", 0) + 1
                    time.sleep(min(30, 3 * self._failures))
        finally:
            self.alive = False
            if awake is not None:
                awake.terminate()
            if self.tunnel:
                self.tunnel.close()
            self.server.close()
            self.store.close()
            self.log("worker stopped")

    def _serve(self, a: dict):
        job, epoch = a["job"], a["epoch"]
        self._neighbours = {side: a[side]["w"] for side in ("prev", "next") if a.get(side)}
        stop = threading.Event()
        streams = job.get("streams", 3)
        senders = {side: Sender(Peer(a[side]["addrs"], a[side]["token"]), streams, cancel=stop.is_set) for side in ("prev", "next") if a.get(side)}

        def watch():
            misses = 0
            while not stop.is_set():
                time.sleep(1)
                try:
                    now = self.coord.json(f"/epoch?w={self.wid}")
                    misses = 0
                    self._prune(now.get("keep_from", -1))
                    if now["epoch"] != epoch or now["finished"]:
                        stop.set()
                except Exception:
                    misses += 1
                    if misses >= 30:
                        self.log("the pool has not answered for 30 s")
                        stop.set()
        threading.Thread(target=watch, daemon=True).start()
        inbox = self.server.inbox
        inbox.drop(lambda k: k[1] == str(epoch))
        seq = a["seq"]
        try:
            while True:
                data = inbox.get(("op", str(epoch), str(seq)), cancel=stop.is_set)
                if data is None:
                    raise Abort()
                op = json.loads(data)
                if op["type"] in ("fb", "fwd"):
                    report = self._pipeline(op, job, epoch, seq, stop, senders)
                elif op["type"] == "commit":
                    report = self._commit(op, job)
                elif op["type"] == "save":
                    report = self._save(op)
                else:
                    raise ValueError(f"unknown operation {op['type']}")
                self.coord.request(f"/done?w={self.wid}&epoch={epoch}&seq={seq}", json.dumps(report).encode())
                seq += 1
        finally:
            stop.set()
            for sender in senders.values():
                sender.close()

    def _commit(self, op: dict, job: dict) -> dict:
        for core in self.chain:
            if op.get("weight_decay") is not None:
                core.set_weight_decay(op["weight_decay"])
            core.step(lr=op.get("lr"), clip=op.get("clip", 1.0), skip=op.get("skip", False))
            if op.get("lower_scale"):
                core.loss_scale = min(core.loss_scale, op["lower_scale"])
        step = op["step"]
        self.state_step = step
        if (step + 1) % job.get("ckpt_every", 4) == 0 or op.get("checkpoint"):
            packed = self._pack(self._snapshot())
            self._keep(step, packed)
            self._upload_later(step, packed)
        return {"step": step}

    def _upload_later(self, step: int, packed: dict):
        """Copies of checkpoints go out one at a time; while one uploads, a newer one replaces the one waiting, so a slow uplink
        sends fewer checkpoints instead of piling them up in memory."""
        with self.store_lock:
            self._next_upload = (self.run_id, step, packed)
            if getattr(self, "_uploading", False):
                return
            self._uploading = True

        def loop():
            while True:
                with self.store_lock:
                    item, self._next_upload = self._next_upload, None
                    if item is None:
                        self._uploading = False
                        return
                if item[0] == self.run_id:
                    self._checkpoint(item[1], item[2], "replica", item[0])
        threading.Thread(target=loop, daemon=True).start()

    def _save(self, op: dict) -> dict:
        if op.get("weights_only"):
            weights = {k: v for k, v in self._snapshot().items() if op.get("moments") or not k.endswith((".m", ".v", ".t"))}
            half = op.get("dtype") in ("bfloat16", "float16")
            by_unit = {}
            for k, v in weights.items():
                by_unit.setdefault(unit_of(k), {})[k] = v
            step = self.state_step

            def send(item):
                u, flat = item
                meta, arrays = (to_half(flat, op["dtype"]) if half else (None, flat))
                self.coord.post_blob(f"/export?w={self.wid}&run={self.run_id}&layer={u}&step={step}", pack(meta, **arrays))
            with ThreadPoolExecutor(4) as ex:
                list(ex.map(send, sorted(by_unit.items())))
            return {"step": step}
        packed = self._pack(self._snapshot())
        self._keep(self.state_step, packed)
        self._checkpoint(self.state_step, packed, op.get("target", "coord"))
        return {"step": self.state_step}

    def _pipeline(self, op: dict, job: dict, epoch: int, seq: int, stop: threading.Event, senders: dict) -> dict:
        n_micro = op["n"]
        backward = op["type"] == "fb"
        inbox = self.server.inbox
        n = len(self.chain)
        clock = {"compute": 0.0, "wait": 0.0}
        per_core = [0.0] * n
        lock = threading.Lock()
        errors = []
        fq = [queue.Queue() for _ in range(n + 1)]
        bq = [queue.Queue() for _ in range(n + 1)]
        for sender in senders.values():
            sender.busy = 0.0
        if not op.get("adapter", True):
            for core in self.chain:
                core.set_adapter(False)

        def timed(kind, fn, *args, ci=None, **kwargs):
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                with lock:
                    clock[kind] += time.time() - t0
                    if ci is not None:
                        per_core[ci] += time.time() - t0

        def blame(side, call):
            try:
                return call()
            except Cancelled:
                raise Abort()
            except ConnectionError as e:
                raise NeighbourDown(a_side(side), e)

        def a_side(side):
            return self._neighbours.get(side)

        def push(side, key, data):
            blame(side, lambda: senders[side].put(key, data))

        def recv(kind, m):
            data = timed("wait", inbox.get, (kind, str(epoch), str(seq), str(m)), cancel=lambda: stop.is_set() or bool(errors))
            if data is None:
                raise Abort()
            return unpack(data)

        def recv_any(kind, pending):
            t0 = time.time()
            try:
                while True:
                    for m in sorted(pending):
                        data = inbox.get((kind, str(epoch), str(seq), str(m)), timeout=0.05)
                        if data is not None:
                            return m, unpack(data)[1]
                    if stop.is_set() or errors:
                        raise Abort()
            finally:
                with lock:
                    clock["wait"] += time.time() - t0

        def get(q):
            t0 = time.time()
            try:
                while True:
                    if stop.is_set() or errors:
                        raise Abort()
                    try:
                        return q.get(timeout=0.2)
                    except queue.Empty:
                        pass
            finally:
                with lock:
                    clock["wait"] += time.time() - t0

        first_stage, last_stage = "prev" not in senders, "next" not in senders
        out_codec = "fp32" if job["codec"] == "fp32" else "fp16"
        grads_back = op.get("grads", False)
        args_of = {}

        def core_args(core, m):
            meta, arrays = args_of[m]
            if not hasattr(core, "device"):
                return mlx_masks(meta, arrays, range(core.first, core.first + len(core.layers))), {}
            return unpack_args(meta, arrays, core.device, core.dtype, range(core.first, core.first + len(core.layers)))

        def grads_message(g, sides, carried=None, core=None):
            parts = with_prefix(encode(g, job["codec"], 0), "g_")
            for k, v in sides.items():
                parts.update(with_prefix(encode(v, "fp32"), f"s{k}_"))
            parts.update(carried_parts(carried or {}, "q", core))
            return pack(None, **parts)

        def carried_parts(carried, letter, core):
            """Block outputs that later blocks take (T5's position bias), or their gradients, in full precision: they can hold
            mask values a 16-bit code cannot."""
            parts = {}
            for (i, k), v in carried.items():
                parts.update(with_prefix(encode(core.numpy(v), "fp32"), f"{letter}{i}_{k}_"))
            return parts

        def carried_from(arrays, letter, core):
            keys = {(int(m.group(1)), int(m.group(2))) for m in (re.match(rf"^{letter}(\d+)_(\d+)_", a) for a in arrays) if m}
            return {key: core.tensor(decode(split_prefix(arrays, f"{letter}{key[0]}_{key[1]}_"))) for key in keys}

        def plan_of(m):
            meta = args_of[m][0] or {}
            return meta.get("carry")

        def add_sides(total, sides, core):
            for k, v in sides.items():
                if v is not None:
                    v = core.numpy(v) if not isinstance(v, np.ndarray) else v
                    total[k] = total[k] + v if k in total else v

        def core_loop(ci):
            core = self.chain[ci]
            head, tail = ci == 0, ci == n - 1
            saved = {}
            for index in range(n_micro):
                if head:
                    m = index
                    meta, arrays = recv("d" if first_stage else "f", m)
                    args_of[m] = ((meta or {}).get("args"), split_prefix(arrays, "a_"))
                    x = core.tensor(decode(split_prefix(arrays, "h_")))
                    carry = carried_from(arrays, "c", core)
                else:
                    m, x, carry = get(fq[ci])
                args, leaves = core_args(core, m)
                plan = plan_of(m)
                kept = {"key": (seq, m), "leaves": leaves} if backward else {}
                if plan:
                    out, onward = timed("compute", core.forward, x, args, carry, plan, ci=ci, **kept)
                else:
                    out, onward = timed("compute", core.forward, x, args, ci=ci, **kept), {}
                if backward:
                    saved[m] = (x, carry)
                if not tail:
                    fq[ci + 1].put((m, out, onward))
                elif last_stage:
                    if op.get("outputs", True):
                        self.coord.send(("o", epoch, seq, m), pack(None, **with_prefix(encode(core.numpy(out), out_codec), "h_")), cancel=stop.is_set)
                else:
                    meta, arrays = args_of[m]
                    push("next", ("f", epoch, seq, m), pack({"args": meta}, **with_prefix(encode(core.numpy(out), job["codec"]), "h_"),
                                                            **with_prefix(arrays, "a_"), **carried_parts(onward, "c", core)))
            if not backward:
                return
            pending = set(saved)
            while pending:
                if tail:
                    m, arrays = recv_any("cg" if last_stage else "b", pending)
                    g = core.tensor(decode(split_prefix(arrays, "g_")))
                    sides = {int(k[1:].split("_", 1)[0]): None for k in arrays if k.startswith("s")}
                    upstream = {k: decode(split_prefix(arrays, f"s{k}_")) for k in sides}
                    carry_grads = carried_from(arrays, "q", core)
                else:
                    m, g, upstream, carry_grads = get(bq[ci + 1])
                pending.discard(m)
                x, carry = saved.pop(m)
                plan = plan_of(m)
                if plan:
                    g_in, sides, back = timed("compute", core.backward, x, None, {}, g, carry, carry_grads, plan, ci=ci, key=(seq, m))
                else:
                    (g_in, sides), back = timed("compute", core.backward, x, None, {}, g, ci=ci, key=(seq, m)), {}
                total = dict(upstream)
                add_sides(total, sides, core)
                if head:
                    if not first_stage:
                        push("prev", ("b", epoch, seq, m), grads_message(core.numpy(g_in), total, back, core))
                    elif grads_back:
                        self.coord.send(("g0", epoch, seq, m), grads_message(core.numpy(g_in), total), cancel=stop.is_set)
                else:
                    bq[ci].put((m, g_in, total, back))

        def guarded(ci):
            try:
                core_loop(ci)
            except Abort:
                errors.append(Abort())
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=guarded, args=(ci,)) for ci in range(n)]
            [th.start() for th in threads]
            [th.join() for th in threads]
            real = [e for e in errors if not isinstance(e, Abort)]
            if real:
                raise real[0]
            if errors or stop.is_set():
                raise Abort()
            for side, sender in senders.items():
                blame(side, sender.flush)
        finally:
            if not op.get("adapter", True):
                for core in self.chain:
                    core.set_adapter(True)
        return {"cores": [[len(c.layers), round(per_core[i], 3), "spill" if getattr(c, "spill", False) else "fast"] for i, c in enumerate(self.chain)], "compute": round(clock["compute"], 3),
                "engine": getattr(self, "engine", None), "wait": round(clock["wait"], 3), "send": round(sum(sd.busy for sd in senders.values()), 3),
                "grad_sq": sum(c.grad_sq() for c in self.chain) if backward else 0.0,
                "loss_scale": min((c.loss_scale for c in self.chain if c.max_loss_scale > 1), default=None)}


def add_worker_args(p: argparse.ArgumentParser):
    p.add_argument("--name", help="how this device shows up in the dashboard (default: host name)")
    p.add_argument("--cap-gb", type=float, help="the most memory layerpool may use on this device, in GB: GPU memory on a machine with NVIDIA or AMD GPUs (all the GPUs it trains on together, shared equally unless --gpu-cap sets each), the shared memory on a Mac, RAM on a computer that trains on its CPU. Default: what it can spare (about 92%% of each GPU, 60%% of a Mac's memory, 50%% of RAM)")
    p.add_argument("--backend", choices=["auto", "torch", "mlx"], default="auto")
    p.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is present")
    p.add_argument("--layers", help="how many of the model's layers this device takes: a count (8), a share (30%%) or a range (4-10); "
                                    "default: as many as suit its memory and its measured speed")
    p.add_argument("--gpus", help="on a machine with several GPUs, the ones to train on, such as 0,2 (default: all)")
    p.add_argument("--gpu-cap", help="the most GPU memory each GPU may use, in GB, such as 0=6,1=10 (GPU numbers as `layerpool devices` "
                                     "lists them); it takes the place of that GPU's share of --cap-gb")
    p.add_argument("--order", type=int, help="this device's place in the chain of devices (lower comes first)")
    p.add_argument("--ram-cap", type=float, help="on a machine with GPUs, the RAM in GB it may also use for layers that do not fit its GPUs; its CPU runs those, more slowly, and layerpool gives it only as many as pay off (default: none)")
    p.add_argument("--allow-sleep", action="store_true", help="let this computer sleep while it is in the pool (by default it stays "
                                                              "awake, since a sleeping device looks gone; closing a laptop's lid still sleeps it)")


def main(argv=None):
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    p = argparse.ArgumentParser(prog="layerpool join", description="Lend this device to a layerpool pool.")
    p.add_argument("coordinator", help="address printed by `layerpool up` or shown by Add device")
    p.add_argument("--token", help="join code from the dashboard or `layerpool invite`; without one, the device asks and waits for approval")
    p.add_argument("--tunnel", choices=["auto", "on", "off"], default="auto",
                   help="open a cloudflared tunnel so other devices can reach this one (auto: when the pool is reached over https or has a tunnel itself)")
    p.add_argument("--identity", help="file that keeps this device's id across restarts (default: ~/.layerpool/device_id)")
    p.add_argument("--port", type=int, default=0, help="port other devices reach this one on (default: any free port)")
    p.add_argument("--advertise", help="address other devices use for this one, such as http://192.168.1.20:7400 (inside Docker, WSL "
                                       "or behind a VPN the address found by itself may be wrong); several are separated by commas")
    add_worker_args(p)
    args = p.parse_args(argv)
    if not args.coordinator.startswith(("http://", "https://")):
        args.coordinator = "http://" + args.coordinator
    tunnel = True if args.tunnel == "on" or args.coordinator.startswith("https://") and args.tunnel == "auto" else (False if args.tunnel == "off" else None)
    worker = None
    try:
        worker = Worker(args.coordinator, args.token, args.name, args.cap_gb, tunnel, port=args.port, backend=args.backend, cpu=args.cpu,
                        identity=device_identity(args.identity), advertise=[a.strip() for a in (args.advertise or "").split(",") if a.strip()],
                        layers=args.layers, gpus=args.gpus, gpu_caps=args.gpu_cap, order=args.order, allow_sleep=args.allow_sleep,
                        ram_cap=args.ram_cap)
        worker.run()
    except KeyboardInterrupt:
        pass
    except ConnectionError:
        if worker is None or getattr(worker, "wid", None) is None:
            sys.exit(f"layerpool: could not reach a pool at {args.coordinator}: check the address that `layerpool up` printed, and "
                     "that both computers are on the same network (or start the pool with --tunnel)")
        sys.exit(f"layerpool: the pool at {args.coordinator} stopped answering, so this device left it")
    except HTTPError as e:
        if e.code != 403:
            sys.exit(f"layerpool: the pool answered {e}")
        if getattr(e, "detail", "").startswith(("the pool denied", "nobody approved")):
            sys.exit(f"layerpool: {e.detail}")
        sys.exit("layerpool: the pool refused this device. The join code may be used up or expired, the pool may have restarted, "
                 "or the device was removed. Get a new join command from the dashboard or `layerpool invite`.")


def machine_key() -> str:
    """What tells this machine apart from others that share its home folder, and stays the same when the host name changes (macOS
    renames it between networks): a hash of the system's machine id, else of the network hardware address, else the host name."""
    found = None
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], capture_output=True, text=True, timeout=5).stdout
            hit = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', out)
            found = hit.group(1) if hit else None
        elif sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                found = winreg.QueryValueEx(key, "MachineGuid")[0]
        else:
            for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        found = f.read().strip() or None
                    break
    except Exception:
        found = None
    if not found:
        node = uuid.getnode()
        found = f"{node:012x}" if not (node >> 40) & 1 else None
    return hashlib.sha256(found.encode()).hexdigest()[:16] if found else socket.gethostname()


def _read_identities(path: str) -> tuple[dict, bool]:
    """The ids kept in that file by machine, and whether it needs writing again: an id from before machines were told apart
    (a bare id, or one under this machine's host name) becomes this machine's."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
    except OSError:
        return {}, False
    try:
        stored = json.loads(text) if text.startswith("{") else {machine_key(): text} if text else {}
    except ValueError:
        return {}, False
    if not isinstance(stored, dict):
        return {}, False
    old = stored.pop(socket.gethostname(), None)
    moved = not text.startswith("{") or (old is not None and machine_key() not in stored)
    if old is not None:
        stored.setdefault(machine_key(), old)
    return stored, moved


def stored_identity(path: str) -> str | None:
    """The id device_identity keeps for this machine in that file, or None, without making one."""
    return _read_identities(path)[0].get(machine_key()) or None


def device_identity(path: str | None = None) -> str:
    """This machine's lasting id, kept in the home folder. It is kept per machine, so machines that share a home folder (a
    network drive, a copied disk image) do not take each other's place in the pool."""
    path = path or os.path.join(os.path.expanduser("~"), ".layerpool", "device_id")
    stored, moved = _read_identities(path)
    key = machine_key()
    if stored.get(key) and not moved:
        return stored[key]
    stored[key] = stored.get(key) or secrets.token_hex(8)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stored, f)
    except OSError:
        pass
    return stored[key]

if __name__ == "__main__":
    main()

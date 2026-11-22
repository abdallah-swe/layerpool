import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings

_macmon = {"proc": None, "last": None}


def _macmon_reader():
    path = shutil.which("macmon")
    if not path:
        return
    try:
        proc = subprocess.Popen([path, "pipe", "-i", "2000"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return
    _macmon["proc"] = proc
    for line in proc.stdout:
        try:
            _macmon["last"] = (time.time(), json.loads(line))
        except ValueError:
            continue


def _mac_extra() -> dict:
    if _macmon["proc"] is None and shutil.which("macmon"):
        threading.Thread(target=_macmon_reader, daemon=True).start()
    got = _macmon["last"]
    if not got or time.time() - got[0] > 10:
        return {}
    data = got[1]
    temp = data.get("temp", {})
    return {"gpu_util": round(100 * float(data.get("gpu_usage", [0, 0])[1]), 1) if isinstance(data.get("gpu_usage"), list) else None,
            "temp_c": temp.get("gpu_temp_avg") or temp.get("cpu_temp_avg"), "power_w": data.get("all_power")}


_heat = {"at": 0.0, "value": None}


def _mac_heat() -> str | None:
    """macOS's heat level, readable without admin rights (the temperature itself is not): "normal", or how far it slowed down."""
    if time.time() - _heat["at"] < 15:
        return _heat["value"]
    _heat["at"] = time.time()
    try:
        text = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    limit = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    if limit and int(limit.group(1)) < 100:
        _heat["value"] = f"hot: slowed to {limit.group(1)}% speed"
    elif "No thermal warning level has been recorded" in text or limit:
        _heat["value"] = "normal (no heat warning)"
    else:
        _heat["value"] = None
    return _heat["value"]


def _cpu_temp() -> float | None:
    """The CPU temperature where the system shares it without admin rights (Linux)."""
    try:
        import psutil
        sensors = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
    except Exception:
        return None
    values = [t.current for name in ("coretemp", "k10temp", "cpu_thermal", "zenpower") for t in sensors.get(name, []) if t.current]
    return round(max(values), 1) if values else None


warnings.filterwarnings("ignore", message="'sin' and 'sout' swap memory stats couldn't be determined", category=RuntimeWarning)


def _ram() -> dict:
    import psutil
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    return {"ram_used_gb": round((vm.total - vm.available) / 1e9, 2), "ram_total_gb": round(vm.total / 1e9, 1), "swap_gb": round(sw.used / 1e9, 2)}


_NVML = {"ready": False}


def _nvml_handles(pynvml) -> list:
    """NVML handles of the GPUs PyTorch uses, in PyTorch's order: NVML counts every GPU by bus, PyTorch only the visible ones,
    fastest first."""
    if not _NVML["ready"]:
        try:
            pynvml.nvmlInit()
        except Exception:
            _NVML["failed"] = True
            raise
        _NVML["ready"] = True
    try:
        import torch
        handles = []
        for i in range(torch.cuda.device_count()):
            uuid = str(torch.cuda.get_device_properties(i).uuid)
            handles.append(pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"))
        if handles:
            return handles
    except Exception:
        pass
    return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]


AMD_SYSFS = "/sys/class/drm"


def _amd() -> list:
    """AMD GPUs from the amdgpu driver's files (Linux), which need no extra package or admin rights."""
    gpus = []
    for total_file in sorted(glob.glob(os.path.join(AMD_SYSFS, "card*", "device", "mem_info_vram_total"))):
        folder = os.path.dirname(total_file)

        def read(name, folder=folder):
            try:
                with open(os.path.join(folder, name), encoding="ascii") as f:
                    return float(f.read().strip())
            except (OSError, ValueError):
                return None
        total, used = read("mem_info_vram_total"), read("mem_info_vram_used")
        if not total:
            continue
        temps = [read(os.path.relpath(path, folder)) for path in glob.glob(os.path.join(folder, "hwmon", "hwmon*", "temp*_input"))]
        temps = [t / 1000 for t in temps if t]
        gpus.append({"used_gb": round((used or 0) / 1e9, 2), "total_gb": round(total / 1e9, 1), "util": read("gpu_busy_percent"),
                     "temp_c": round(max(temps), 1) if temps else None, "power_w": None})
    return gpus


def _cuda() -> dict:
    out = {"gpus": []}
    try:
        import torch
        if getattr(torch.version, "hip", None) and sys.platform.startswith("linux"):
            out["gpus"] = _amd()
            if out["gpus"]:
                return out
    except Exception:
        pass
    try:
        if _NVML.get("failed"):
            raise RuntimeError("no NVML here")
        import pynvml
        for h in _nvml_handles(pynvml):
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpu = {"used_gb": round(mem.used / 1e9, 2), "total_gb": round(mem.total / 1e9, 1)}
            for key, fn in (("util", lambda: pynvml.nvmlDeviceGetUtilizationRates(h).gpu), ("temp_c", lambda: pynvml.nvmlDeviceGetTemperature(h, 0)),
                            ("power_w", lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000)):
                try:
                    gpu[key] = round(float(fn()), 1)
                except Exception:
                    gpu[key] = None
            out["gpus"].append(gpu)
        return out
    except Exception:
        pass
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            out["gpus"].append({"used_gb": round((total - free) / 1e9, 2), "total_gb": round(total / 1e9, 1), "util": None, "temp_c": None, "power_w": None})
    except Exception:
        pass
    return out


def sample(backend: str, cpu: bool = False) -> dict:
    out = {"time": round(time.time(), 1), **_ram()}
    if backend == "mlx" and not cpu:
        try:
            import mlx.core as mx
            out["mlx_active_gb"] = round(mx.get_active_memory() / 1e9, 2)
            out["mlx_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
            out["mlx_cache_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass
        out.update({k: v for k, v in _mac_extra().items() if v is not None})
    elif backend == "torch" and not cpu:
        out.update(_cuda())
    if out.get("temp_c") is None and not any(g.get("temp_c") for g in out.get("gpus", [])):
        if sys.platform == "darwin":
            heat = _mac_heat()
            if heat:
                out["heat"] = heat
        else:
            cpu_temp = _cpu_temp()
            if cpu_temp is not None:
                out["temp_c"] = cpu_temp
    return out

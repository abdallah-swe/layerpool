import hmac
import json
import math
import os
import queue
import re
import secrets
import shutil
import tempfile
import threading
import time

import numpy as np

from . import __version__
from .codec import Spilled, from_half, pack, replace_file, unpack
from .export import export_blocks, export_peft
from .net import Cancelled, Peer, PeerServer, ping
from .planner import Node, Stage, layer_bounds, plan
from .tunnel import Tunnel

DEFAULT_SPEED = {"cuda": 1.0, "mlx": 0.6, "mps": 1.5, "cpu": 8.0}
OPLOG_LIMIT_GB = float(os.environ.get("LAYERPOOL_REPLAY_GB", "4"))
CKPT_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "layerpool", "checkpoints")
SHARED_MEMORY = {"cpu": 0.5, "mlx": 0.6, "mps": 0.6}


OPTIMIZER_FILE = "layerpool_optimizer.safetensors"

def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _version_tuple(text: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", str(text).split("+")[0])[:3])


def _transformers_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("transformers")
    except PackageNotFoundError:
        return ""


class Unreachable(ConnectionError):
    def __init__(self, wid: str, error: Exception):
        super().__init__(f"{wid}: {error!r}")
        self.wid = wid


class Interrupted(Exception):
    pass


def _clean_telemetry(data) -> dict:
    """Keep only numbers (and lists of GPU records with numbers) from a device's report, plus a short heat note, so a bad device
    cannot break the dashboard."""

    def number(v):
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
    out = {}
    for key, value in (data or {}).items() if isinstance(data, dict) else []:
        key = str(key)[:32]
        if key == "gpus" and isinstance(value, list):
            out[key] = [{str(k)[:32]: number(v) for k, v in g.items()} for g in value[:16] if isinstance(g, dict)]
        elif number(value) is not None:
            out[key] = number(value)
        elif key == "heat" and isinstance(value, str):
            out[key] = "".join(c for c in value[:60] if c.isprintable())
    return out


class PoolError(RuntimeError):
    pass


DEVICE_SETTINGS = ("name", "cap_gb", "enabled", "layers", "gpus", "gpu_caps", "order", "ram_cap")


def clean_settings(changes: dict, gpu_count: int) -> dict:
    """A device's settings checked and in one form, or PoolError saying what is wrong. None (or "auto") clears a setting.
    layers: a count ("8"), a share of the model ("30%") or a range ("4-10"); gpus: the GPUs to use on a machine with several
    ([0, 2]); gpu_caps: the most memory each GPU may use ({"0": 6, "1": 10}); order: the device's place in the chain (lower first);
    ram_cap: on a machine with GPUs, the RAM it may also use for layers that do not fit them, which its CPU then runs."""
    bad = set(changes) - set(DEVICE_SETTINGS)
    if bad:
        raise PoolError(f"unknown device settings: {sorted(bad)}")
    out = {}
    for key, value in changes.items():
        if value == "auto":
            value = None
        if key == "ram_cap" and value is not None and not gpu_count:
            raise PoolError("ram_cap lets a machine with GPUs keep more layers in its RAM; on this device the memory cap already is its RAM")
        if key in ("cap_gb", "ram_cap") and value is not None:
            if isinstance(value, bool) or not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise PoolError(f"{key} must be a positive number")
        elif key == "name":
            value = str(value or "").strip()[:64]
            if not value:
                raise PoolError("a device name cannot be empty")
        elif key == "enabled":
            value = bool(value)
        elif key == "layers" and value is not None:
            value = str(value).replace(" ", "")
            if not re.fullmatch(r"[1-9]\d{0,3}|[1-9]\d{0,3}-[1-9]\d{0,3}|(100|[1-9]\d?(\.\d+)?)%", value):
                raise PoolError("layers takes a count (8), a share of the model (30%), a range (4-10) or auto")
            low, high = layer_bounds(value, 100)
            if high is not None and low > high:
                raise PoolError(f"layers {value}: the range goes the wrong way")
        elif key == "gpus" and value is not None:
            if isinstance(value, str):
                value = [x for x in value.replace(" ", "").split(",") if x]
            try:
                value = sorted({int(x) for x in value})
            except (TypeError, ValueError):
                raise PoolError("gpus takes GPU numbers such as 0,2") from None
            if not value or any(i < 0 or i >= gpu_count for i in value):
                raise PoolError(f"gpus: this device has {gpu_count} GPU{'s' if gpu_count != 1 else ''}, numbered 0 to {max(0, gpu_count - 1)}")
        elif key == "gpu_caps" and value is not None:
            if isinstance(value, str):
                try:
                    value = dict(part.split("=", 1) for part in value.replace(" ", "").split(",") if part)
                except ValueError:
                    raise PoolError("gpu_caps takes GPU=GB pairs such as 0=6,1=10") from None
            try:
                value = {str(int(k)): float(v) for k, v in dict(value).items()}
            except (TypeError, ValueError):
                raise PoolError("gpu_caps takes GPU=GB pairs such as 0=6,1=10") from None
            if any(int(k) < 0 or int(k) >= gpu_count for k in value) or any(not (math.isfinite(v) and v > 0) for v in value.values()):
                raise PoolError(f"gpu_caps: GPU numbers go from 0 to {max(0, gpu_count - 1)} on this device and each cap is a positive "
                                "number of GB")
            value = value or None
        elif key == "order" and value is not None:
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise PoolError("order takes a whole number (lower comes first in the chain)") from None
        out[key] = value
    return out


def scheduled_lr(base: float, step: int, total: int, warmup: int = 0, schedule: str = "constant", min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    if schedule == "cosine":
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
    if schedule == "linear":
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return base * (1 - (1 - min_ratio) * progress)
    return base


def _thin(rows: list, most: int) -> list:
    """At most `most` rows spread over the whole list, the last one always kept, for charts of long runs."""
    if len(rows) <= most:
        return list(rows)
    step = len(rows) / (most - 1)
    return [rows[int(i * step)] for i in range(most - 1)] + [rows[-1]]


class Pool:
    """Devices join once and stay; runs come and go. Every surface (library, CLI, dashboard) drives runs through these methods."""

    def __init__(self, port: int = 0, heartbeat_timeout: float = 120.0, tunnel: bool = False, log=None, ckpt_dir: str | None = None,
                 max_failures: int = 3, token: str | None = None, settings_file: str | None = None,
                 state_file: str | None = None, advertise: str | None = None):
        self.heartbeat_timeout = heartbeat_timeout
        self.state_file = state_file
        kept = {}
        if state_file and os.path.exists(state_file):
            try:
                with open(state_file, encoding="utf-8") as f:
                    kept = json.load(f)
            except ValueError:
                kept = {}
        token = token or kept.get("token")
        self.settings_file = settings_file
        self.saved_settings: dict[str, dict] = {}
        if settings_file and os.path.exists(settings_file):
            try:
                with open(settings_file, encoding="utf-8") as f:
                    self.saved_settings = json.load(f)
            except ValueError:
                pass
        self.cond = threading.Condition()
        self.op_lock = threading.Lock()
        self.workers: dict[str, dict] = {}
        self.identities: dict[str, str] = dict(kept.get("identities", {}))
        self.plan: list[dict] | None = None
        self.epoch = 0
        self.step = 0
        self.seq = 0
        self.seq_base = 0
        self.ready: set = set()
        self.reports: dict[tuple, dict] = {}
        self.measured: dict[str, float] = {}
        self.measured_link: dict[str, float] = {}
        self.measured_spill: dict[str, float] = {}
        self.told_bnb: set = set()
        self.steps_in_plan = 0
        self.pending = False
        self.closed = False
        self.job: dict | None = None
        self._planning = threading.local()
        self.run_id: str | None = None
        self.runs: list[dict] = []
        self.oplog: list[dict] = []
        self.oplog_after = -1
        self._replayed_epoch = -1
        self._good_steps = 0
        self.accum = {"loss": None, "tokens": 0, "started": None}
        self._own_ckpt = ckpt_dir is None
        if ckpt_dir is None:
            os.makedirs(CKPT_ROOT, exist_ok=True)
            ckpt_dir = tempfile.mkdtemp(prefix="pool_", dir=CKPT_ROOT)
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt: dict[int, dict[int, set[str]]] = {}
        self.restore = -1
        self.protect: int | None = None
        self.max_failures = max_failures
        self.history: list[dict] = []
        self.events: list[tuple[float, str]] = []
        self.listeners: list = []
        self.t0 = time.time()
        self.log_fn = log or (lambda text: print(f"[layerpool {time.time() - self.t0:7.1f}s] {text}", flush=True))
        routes = {
            ("POST", "/join"): self._join, ("POST", "/hb"): self._hb, ("GET", "/assignment"): self._assignment, ("GET", "/epoch"): self._epoch,
            ("POST", "/ready"): self._ready, ("POST", "/done"): self._done, ("POST", "/ckpt"): self._ckpt_put, ("GET", "/ckpt"): self._ckpt_get,
            ("GET", "/info"): lambda q, b: (200, {"tunnel": self.tunnel is not None, "version": __version__}), ("POST", "/export"): self._export_put,
            ("POST", "/fail"): self._fail, ("POST", "/have"): self._have, ("GET", "/where"): self._where,
            ("POST", "/request"): self._request_join, ("GET", "/request"): self._request_status,
        }
        self.requests: dict[str, dict] = {}
        self.invites: dict[str, dict] = {}
        self.device_tokens: dict[str, str] = dict(kept.get("device_tokens", {}))
        self.server = PeerServer(port=port, token=token, routes=routes, check=self._check,
                                 limits={"/request": 8192, "/join": 1 << 16, "/hb": 1 << 20, "/info": 1024})
        self.token = self.server.token
        self.urls = [self.server.url_at(advertise) if advertise else self.server.lan_url()]
        self.tunnel = None
        if tunnel:
            opened = queue.Queue()
            threading.Thread(target=self._keep_tunnel, args=(opened,), daemon=True).start()
            got = opened.get()
            if isinstance(got, BaseException):
                self.server.close()
                raise got
            self.tunnel = got
            self.urls.append(self.tunnel.url)
        threading.Thread(target=self._monitor, daemon=True).start()

    def _keep_tunnel(self, opened):
        """Opens the tunnel, watches it for as long as the pool runs, and opens new ones from here: on Linux a helper process ends
        with the thread that started it, so this thread must outlive every tunnel it opens."""
        try:
            opened.put(Tunnel(self.server.port, log=self.event))
        except BaseException as e:
            opened.put(e)
            return
        while not self.closed:
            time.sleep(1)
            if self.tunnel is not None and not self.tunnel.alive() and not self.closed:
                self._reopen_tunnel()

    def _reopen_tunnel(self):
        """cloudflared stopped: open a new tunnel. It has a new address, so devices on the internet must join again with a new
        join command; the ones on this network go on."""
        old = self.tunnel.url
        self.event("the tunnel stopped; opening a new one")
        try:
            self.tunnel = Tunnel(self.server.port, log=self.event)
        except Exception as e:
            self.event(f"could not open a new tunnel ({str(e).splitlines()[0]}); trying again in a minute")
            time.sleep(60)
            return
        self.urls = [u for u in self.urls if u != old] + [self.tunnel.url]
        self.event(f"new tunnel at {self.tunnel.url}: devices on the internet need a new join command (dashboard or `layerpool invite`)")

    def event(self, text: str):
        if getattr(self._planning, "previewing", False):
            return
        item = (round(time.time() - self.t0, 1), text)
        self.events.append(item)
        self.events[:] = self.events[-500:]
        self.log_fn(text)
        for listener in list(self.listeners):
            listener("event", {"t": item[0], "text": text})

    def emit(self, kind: str, data: dict):
        for listener in list(self.listeners):
            listener(kind, data)

    def _seen(self, query):
        w = query.get("w")
        if w in self.workers:
            info = self.workers[w]
            info["seen"] = time.time()
            if not info["alive"] and not info.get("banned") and not info.get("removed"):
                info["alive"] = True
                self.pending = True
                self.event(f"{info['name']} is back")

    @staticmethod
    def _clean_join(info: dict) -> dict:
        """Everything a device says about itself ends up in the dashboard and the plan, so keep only well-formed values."""
        def number(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return v if 0 < v < 1e6 else None
        text = lambda v, n=80: "".join(c for c in str(v)[:n] if c.isprintable()) if v is not None else ""
        devices = [{"kind": text(d.get("kind"), 16), "name": text(d.get("name")), "gb": number(d.get("gb")) or 0.0, "amd": bool(d.get("amd")),
                    "q4": bool(d.get("q4", not d.get("amd")))}
                   for d in (info.get("devices") or [])[:16] if isinstance(d, dict)] or [{"kind": "cpu", "name": "unknown", "gb": 0.0}]
        addrs = [a for a in (info.get("addrs") or [])[:4] if isinstance(a, str) and a.startswith(("http://", "https://")) and len(a) < 200]
        libraries = {text(k, 24): text(v, 40) for k, v in list((info.get("libraries") or {}).items())[:12]} if isinstance(info.get("libraries"), dict) else {}
        return {**info, "libraries": libraries, "name": text(info.get("name"), 40) or "device", "cap_gb": number(info.get("cap_gb")),
                "devices": devices, "addrs": addrs, "backend": text(info.get("backend"), 16), "token": text(info.get("token"), 128),
                "identity": text(info.get("identity"), 64) or None, "version": text(info.get("version"), 32)}

    def _join(self, query, body):
        info = self._clean_join(json.loads(body))
        gpu_count = sum(1 for d in info["devices"] if d["kind"] == "cuda")
        asked = {k: info.pop(k) for k in ("layers", "gpus", "gpu_caps", "order", "ram_cap") if info.get(k) is not None}
        if info.get("cap_gb") is not None:
            asked["cap_gb"] = info["cap_gb"]
        try:
            info["asked"] = clean_settings(asked, gpu_count)
        except PoolError as e:
            return 400, {"error": str(e)}
        with self.cond:
            if info.get("version") and info["version"].split(".")[:2] != __version__.split(".")[:2]:
                return 409, {"error": f"this pool runs layerpool {__version__}, the device has {info['version']}: run pip install -U layerpool=={__version__}"}
            theirs = str((info.get("libraries") or {}).get("transformers") or "")
            ours = _transformers_version()
            if theirs and ours and theirs.split(".")[0] != ours.split(".")[0]:
                return 409, {"error": f"this pool runs transformers {ours} and the device has {theirs}: install the same major version on both, "
                                      f"for example pip install \"transformers=={ours}\""}
            if theirs and ours and theirs.split(".")[:2] != ours.split(".")[:2]:
                self.event(f"note: {info['name']} has transformers {theirs} and this pool {ours}; the same version everywhere is safest")
            identity = info.get("identity")
            wid = self.identities.get(identity) if identity else None
            if wid is None or wid not in self.workers or self.workers[wid].get("removed"):
                if wid is None or self.workers.get(wid, {}).get("removed"):
                    used = set(self.workers) | set(self.identities.values())
                    wid = next(f"w{k}" for k in range(len(used) + 1) if f"w{k}" not in used)
                if identity:
                    self.identities[identity] = wid
                settings = {"version": 1, "name": info["name"], "cap_gb": None, "enabled": True}
                for key, value in (self.saved_settings.get(identity) or {}).items() if identity else ():
                    try:
                        settings.update(clean_settings({key: value}, gpu_count))
                    except PoolError as e:
                        self.event(f"{info['name']}: a saved setting no longer fits this device and is dropped ({e})")
                settings.update(info.pop("asked", {}))
                settings["version"] = 2
            else:
                settings = self.workers[wid]["settings"]
                if info.get("asked"):
                    settings.update(info["asked"])
                    settings["version"] = settings.get("version", 0) + 1
                info.pop("asked", None)
                for steps in self.ckpt.values():
                    for holders in steps.values():
                        holders.discard(wid)
            self.workers[wid] = {**info, "name": settings.get("name") or info["name"], "cap_gb": settings.get("cap_gb"), "alive": True,
                                 "seen": time.time(), "fails": 0, "peer": Peer(info["addrs"], info["token"]), "settings": settings, "telemetry": {},
                                 "joined": time.time()}
            self._remember(self.workers[wid])
            self.event(f"{self.workers[wid]['name']} joined as {wid} with {[d['name'] for d in info['devices']]}")
            threading.Thread(target=self._check_reachable, args=(wid, list(info["addrs"])), daemon=True).start()
            if self.job is not None:
                if self.plan is None:
                    self._replan(f"{info['name']} joined", lost=False)
                elif any(p["w"] == wid for p in self.plan):
                    self._replan(f"{info['name']} restarted and lost its layers", lost=True)
                else:
                    self.pending = True
            self.cond.notify_all()
        device_token = secrets.token_hex(16)
        with self.cond:
            for t, w in [(t, w) for t, w in self.device_tokens.items() if w == wid]:
                del self.device_tokens[t]
            self.device_tokens[device_token] = wid
            used = query.get("_invite")
            if used in self.invites and not self.invites[used]["reusable"]:
                del self.invites[used]
            self._save_state()
        return 200, {"wid": wid, "token": device_token, "heartbeat": max(0.5, min(5.0, self.heartbeat_timeout / 4)), "settings": self.workers[wid]["settings"]}

    REQUEST_TTL = 900

    def _request_join(self, query, body):
        """A device without a join code asks to join. It waits until someone at the pool approves its code."""
        if len(body) > 8192:
            return 413, {"error": "request too large"}
        try:
            info = json.loads(body)
        except ValueError:
            return 400, {"error": "bad request"}
        text = lambda v, n: "".join(c for c in str(v or "")[:n] if c.isprintable())
        with self.cond:
            now = time.time()
            for t in [t for t, r in self.requests.items() if now - r["time"] > self.REQUEST_TTL]:
                del self.requests[t]
            if sum(r["status"] == "waiting" for r in self.requests.values()) >= 20:
                return 429, {"error": "too many devices are waiting for approval; approve or deny them first"}
            alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
            code = "".join(secrets.choice(alphabet) for _ in range(6))
            ticket = secrets.token_urlsafe(24)
            self.requests[ticket] = {"code": code, "name": text(info.get("name"), 40) or "device", "ip": str(query.get("_ip", ""))[:64],
                                     "hardware": [text(d, 80) for d in (info.get("devices") or [])[:8]], "time": now, "status": "waiting"}
        self.event(f"{self.requests[ticket]['name']} asks to join (code {code}): approve it in the dashboard or with `layerpool approve {code}`")
        return 200, {"code": code, "ticket": ticket, "expires_in_s": self.REQUEST_TTL}

    def _request_status(self, query, body):
        with self.cond:
            req = self.requests.get(str(query.get("ticket", "")))
            if req is None:
                return 404, {"error": "this request is unknown or expired; ask again"}
            if req["status"] == "approved":
                if "invite" not in req:
                    req["invite"] = self.invite(ttl=self.REQUEST_TTL)["code"]
                return 200, {"status": "approved", "invite": req["invite"]}
            return 200, {"status": req["status"]}

    def waiting(self) -> list[dict]:
        """Devices waiting for approval."""
        with self.cond:
            now = time.time()
            return [{"code": r["code"], "name": r["name"], "hardware": r["hardware"], "from": r.get("ip", ""), "waiting_s": round(now - r["time"])}
                    for r in self.requests.values() if r["status"] == "waiting" and now - r["time"] <= self.REQUEST_TTL]

    def answer(self, code: str, approve: bool = True) -> str:
        """Approve or deny a device that asked to join; returns its name."""
        code = str(code).strip().upper().replace("-", "")
        with self.cond:
            for r in self.requests.values():
                if r["code"] == code and r["status"] == "waiting":
                    r["status"] = "approved" if approve else "denied"
                    name = r["name"]
                    break
            else:
                raise KeyError(f"no device is waiting with code {code}")
        self.event(f"{name} was {'approved' if approve else 'denied'}")
        return name

    def _save_state(self):
        if not self.state_file:
            return
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            fd = os.open(self.state_file + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"token": self.token, "device_tokens": self.device_tokens, "identities": self.identities}, f)
            replace_file(self.state_file + ".tmp", self.state_file)
        except OSError:
            pass

    def _check_reachable(self, wid: str, addrs: list[str]):
        """Devices send their work straight to each other: say so plainly when the pool cannot reach a device that joined."""
        for attempt in range(7):
            if not addrs or any(ping(a) for a in addrs) or self.closed or wid not in self.workers:
                return
            time.sleep(10)
        name = self.workers.get(wid, {}).get("name", wid)
        self.event(f"{name} joined, but this pool cannot reach it at {', '.join(addrs)}: other devices may not reach it either. "
                   "Allow layerpool through its firewall, give it an address with --advertise, or join with --tunnel on")

    def _hb(self, query, body):
        with self.cond:
            self._seen(query)
            info = self.workers.get(query.get("w"))
            if info is None:
                return 404, b""
            if body:
                data = json.loads(body)
                info["telemetry"] = _clean_telemetry(data.get("telemetry"))
                applied = data.get("applied", 0)
                info["applied"] = applied if isinstance(applied, int) else 0
            return 200, {"settings": info["settings"]}

    def _assignment(self, query, body):
        with self.cond:
            self._seen(query)
            if self.closed:
                if query.get("w") in self.workers:
                    self.workers[query["w"]]["bye"] = True
                    self.cond.notify_all()
                return 200, {"done": True}
            stages = self.plan or []
            if self.job is not None:
                for i, stage in enumerate(stages):
                    if stage["w"] == query["w"]:
                        neighbour = lambda j: None if j < 0 or j >= len(stages) else {
                            "w": stages[j]["w"], "addrs": self.workers[stages[j]["w"]]["addrs"], "token": self.workers[stages[j]["w"]]["token"]}
                        return 200, {"run": self.run_id, "epoch": self.epoch, "seq": self.seq_base, "range": stage["range"], "on_gpu": stage.get("on_gpu"), "step": self.step,
                                     "restore": self.restore, "prev": neighbour(i - 1), "next": neighbour(i + 1), "replica": self._replica_target(query["w"]),
                                     "keep_from": self.keep_from(), "job": self.job, "settings": self.workers[query["w"]]["settings"]}
            info = self.workers.get(query.get("w"), {})
            return 200, {"wait": True, "run": self.run_id if self.job is not None else None, "left_out": info.get("why_left_out") if info.get("banned") else None}

    def _epoch(self, query, body):
        with self.cond:
            self._seen(query)
            return 200, {"epoch": self.epoch, "finished": self.closed, "keep_from": self.keep_from()}

    def _ready(self, query, body):
        with self.cond:
            self._seen(query)
            self.ready.add((query["w"], int(query["epoch"])))
            self.cond.notify_all()
        return 200, b""

    def _done(self, query, body):
        with self.cond:
            self._seen(query)
            self.reports[(query["w"], int(query["epoch"]), int(query["seq"]))] = json.loads(body) if body else {}
            self.cond.notify_all()
        return 200, b""

    def _answers(self, wid: str) -> bool:
        return any(ping(url, 10) for url in self.workers[wid]["addrs"])

    def _lose(self, wid: str, why: str):
        info = self.workers[wid]
        if info["alive"]:
            info["alive"] = False
            self.event(f"{info['name']} {why}")
            if self.job is not None and any(p["w"] == wid for p in self.plan or []):
                self._replan(f"{info['name']} dropped out")

    def _fail(self, query, body):
        blame = query.get("blame")
        if blame in self.workers and not self._answers(blame):
            with self.cond:
                self._seen(query)
                self._lose(blame, f"does not answer ({self.workers[query['w']]['name']} could not reach it)")
            return 200, b""
        with self.cond:
            self._seen(query)
            info = self.workers[query["w"]]
            if query.get("switched") and self.job is not None:
                info["no_mlx"] = self.job["repo"]
                self._replan(f"{info['name']} trains this model with PyTorch instead of MLX, which failed on it: {body.decode()[:200]}")
                return 200, b""
            oom = _number(query.get("oom"))
            room = info.get("cap_gb") or sum(d["gb"] for d in info["devices"])
            if oom is not None and room > 1.0:
                cap = round(max(0.5, 0.85 * min(room, oom or room)), 1)
                info["cap_gb"], info["oom_capped"] = cap, True
                why = f"{info['name']} ran out of memory; it now gets layers for at most {cap} GB (set its cap in the dashboard to change this)"
                if self.job is not None:
                    self._replan(why)
                else:
                    self.event(why)
                return 200, b""
            info["fails"] += 1
            why = f"{info['name']} reported an error: {body.decode()[:300]}"
            if info["fails"] >= self.max_failures:
                info["alive"], info["banned"] = False, True
                info["why_left_out"] = body.decode()[:300]
                why += f" ({info['fails']} times in a row, it is left out)"
            if self.job is not None:
                self._replan(why)
            else:
                self.event(why)
        return 200, b""

    def _units(self) -> list[int]:
        return list(range(self.job["layers"]))

    def _path(self, unit: int, step: int) -> str:
        return os.path.join(self.ckpt_dir, f"{self.run_id}_{unit}_{step}.npz")

    def _available(self, unit: int, step: int) -> bool:
        holders = self.ckpt.get(unit, {}).get(step, set())
        return any(h == "coord" or (h in self.workers and self.workers[h]["alive"]) for h in holders)

    def complete_step(self) -> int:
        if self.job is None:
            return -1
        steps = None
        for unit in self._units():
            got = {st for st in self.ckpt.get(unit, {}) if self._available(unit, st)}
            steps = got if steps is None else steps & got
        return max(steps) if steps else -1

    def keep_from(self) -> int:
        done = self.complete_step()
        return done if self.protect is None else min(done, self.protect)

    def _replica_target(self, wid: str) -> dict | None:
        mode = self.job.get("peer_replicas", "auto")
        if mode == "never" or not self.plan:
            return None
        ring = [p["w"] for p in self.plan if mode == "always" or not self.workers[p["w"]].get("near")]
        if wid not in ring or len(ring) < 2:
            return None
        target = ring[(ring.index(wid) + 1) % len(ring)]
        return {"w": target, "addrs": self.workers[target]["addrs"], "token": self.workers[target]["token"]}

    def _prune(self):
        low = self.keep_from()
        for unit, steps in self.ckpt.items():
            for old in [st for st in steps if st < low]:
                if "coord" in steps.pop(old):
                    try:
                        os.remove(self._path(unit, old))
                    except FileNotFoundError:
                        pass
        floor = self.keep_from()
        if floor > self.oplog_after:
            self._set_oplog([op for op in self.oplog if op["step"] > floor])
            self.oplog_after = floor

    def _add(self, unit: int, step: int, holder: str):
        self.ckpt.setdefault(unit, {}).setdefault(step, set()).add(holder)

    def _ckpt_put(self, query, body):
        if query.get("run") and query["run"] != self.run_id:
            return 409, {"error": "this checkpoint belongs to a run that has ended"}
        unit, step = int(query["layer"]), int(query["step"])
        path = self._path(unit, step)
        with open(path + ".part", "wb") as f:
            f.write(body)
        replace_file(path + ".part", path)
        with self.cond:
            self._add(unit, step, "coord")
            self._prune()
            self.cond.notify_all()
        return 200, b""

    def _export_put(self, query, body):
        if query.get("run") and query["run"] != self.run_id:
            return 409, {"error": "this export belongs to a run that has ended"}
        path = self._path(int(query["layer"]), int(query["step"])) + ".weights"
        with open(path + ".part", "wb") as f:
            f.write(body)
        replace_file(path + ".part", path)
        return 200, b""

    def _have(self, query, body):
        info = json.loads(body)
        if query.get("run") and query["run"] != self.run_id:
            return 409, {"error": "this checkpoint belongs to a run that has ended"}
        with self.cond:
            self._seen(query)
            for unit in info["layers"]:
                for holder in info["holders"]:
                    if holder in self.workers:
                        self._add(int(unit), int(info["step"]), holder)
            self._prune()
            self.cond.notify_all()
        return 200, b""

    def _where(self, query, body):
        unit, step, epoch = int(query["layer"]), int(query["step"]), int(query.get("epoch", -1))
        with self.cond:
            self.cond.wait_for(lambda: self._available(unit, step) or (epoch >= 0 and self.epoch != epoch), timeout=float(query.get("wait", 60)))
            if epoch >= 0 and self.epoch != epoch:
                return 409, b""
            holders = self.ckpt.get(unit, {}).get(step, set())
            out = [{"w": "coord"}] if "coord" in holders else []
            out += [{"w": h, "addrs": self.workers[h]["addrs"], "token": self.workers[h]["token"]} for h in sorted(holders)
                    if h in self.workers and self.workers[h]["alive"] and h != query.get("w")]
            if query.get("w") in holders:
                out.insert(0, {"w": query["w"]})
            return (200, {"holders": out}) if out else (404, b"")

    def _ckpt_get(self, query, body):
        unit, step = int(query["layer"]), int(query["step"])
        with self.cond:
            if "coord" not in self.ckpt.get(unit, {}).get(step, set()):
                return 404, b""
        with open(self._path(unit, step), "rb") as f:
            return 200, f.read()

    def _job(self) -> dict | None:
        """The job the split is worked out for: a run that plan_for asks about, on its own thread only, or the running one."""
        return getattr(self._planning, "job", None) or self.job

    def _nodes(self, alive: list[str]) -> list[Node]:
        job, nodes = self._job(), []
        usable = self._usable(alive)
        left_out = [w for w in alive if w not in usable]
        told = self.__dict__.setdefault("_told_q4", set())
        fresh = [w for w in left_out if (self.run_id, w) not in told and (self.run_id, w) not in self.told_bnb]
        if fresh and not getattr(self._planning, "previewing", False):
            told.update((self.run_id, w) for w in fresh)
            names = ", ".join(self.workers[w]["name"] for w in fresh)
            self.event(f"{names} sit{'s' if len(fresh) == 1 else ''} this run out: a bitsandbytes 4-bit model needs a GPU bitsandbytes "
                       f"runs on (NVIDIA, or a recent AMD one), or an MLX 4-bit version (mlx_model=) on Apple Silicon")
        for w in usable:
            alive = usable
            info = self.workers[w]
            settings = info["settings"]
            kind = info["devices"][0]["kind"]
            devices = [(i, d) for i, d in enumerate(info["devices"]) if kind != "cuda" or not settings.get("gpus") or i in settings["gpus"]]
            cap_gb, gpu_caps = info.get("cap_gb"), settings.get("gpu_caps") or {}
            share = cap_gb / len(devices) if cap_gb else 1e9
            cap = sum(max(0.0, min(d["gb"] * 0.92 - 0.6, gpu_caps.get(str(i), share))) for i, d in devices)
            if cap_gb:
                cap = min(cap, cap_gb)
            if kind in SHARED_MEMORY:
                cap = min(cap, cap_gb or info["devices"][0]["gb"] * SHARED_MEMORY[kind])
            guess = self._guess(w)
            speed = self.measured.get(w) or guess * self._calibration(alive)
            engine = self._engine(info)
            sizes = {**job, **job.get("sizes", {}).get("mlx" if engine == "mlx" else "torch", {})}
            fixed = sizes.get("overhead_gb", 0.5)
            driver = job.get("driver") or {}
            if info.get("identity") and info["identity"] in driver.get("identities", []):
                fixed += driver.get("gb", 0.0)
            per_layer = sizes.get("layer_gb", 0.0) * (2.0 if engine == "cpu" and job.get("mode", "lora") != "full" else 1.0)
            spill = {}
            if engine == "cuda" and settings.get("ram_cap") and job.get("kind") != "q4":
                cpu_speed = self.measured_spill.get(w) or DEFAULT_SPEED["cpu"] * self._calibration(alive)
                cpu_layer = sizes.get("layer_gb", 0.0) * (1.0 if job.get("mode", "lora") == "full" else 2.0)
                spill = {"spill_gb": settings["ram_cap"], "spill_layer_gb": cpu_layer, "spill_sec_per_layer": cpu_speed}
            low, high = layer_bounds(settings.get("layers"), job["layers"])
            if low and info.get("oom_capped") and fixed + low * per_layer > cap:
                fits = max(0, int((cap - fixed) / per_layer)) if per_layer else low
                if not getattr(self._planning, "previewing", False) and info.get("told_oom_layers") != (self.run_id, fits):
                    info["told_oom_layers"] = (self.run_id, fits)
                    self.event(f"{info['name']} ran out of memory with {low} layers, so it takes {fits} until its memory cap is set again")
                low, high = min(low, fits), min(high or fits, fits)
            nodes.append(Node(w, cap, fixed, speed, layer_gb=per_layer, link_sec=self.measured_link.get(w, 0.0), min_layers=low, max_layers=high,
                              **spill))
        return nodes if usable else []

    def _usable(self, alive: list[str]) -> list[str]:
        job = self._job()
        kind = lambda w: self.workers[w]["devices"][0]["kind"]
        if job["kind"] != "q4":
            return list(alive)
        return [w for w in alive if (kind(w) == "cuda" and self.workers[w]["devices"][0].get("q4", True) and self._has_bnb(w))
                or (kind(w) == "mlx" and job.get("mlx_repo"))]

    def _tell_slow_kernels(self, workers: list[str]):
        """A model with linear-attention blocks (Qwen3.5, Qwen3-Next) runs them far faster on NVIDIA GPUs with
        flash-linear-attention and causal-conv1d; devices without them work, slowly, so the events say which."""
        if not (self.job or {}).get("linear_attention"):
            return
        told = self.__dict__.setdefault("_told_kernels", set())
        slow = []
        for w in workers:
            info = self.workers.get(w) or {}
            libraries = info.get("libraries") or {}
            if info.get("devices", [{}])[0].get("kind") == "cuda" and not info["devices"][0].get("amd") and "transformers" in libraries \
                    and not ("flash-linear-attention" in libraries and "causal-conv1d" in libraries) and (self.run_id, w) not in told:
                told.add((self.run_id, w))
                slow.append(info.get("name", w))
        if slow:
            self.event(f"{', '.join(slow)} run{'s' if len(slow) == 1 else ''} this model's linear-attention blocks without their fast "
                       "kernels, which is much slower: run pip install flash-linear-attention causal-conv1d there and join again")

    def _has_bnb(self, w: str) -> bool:
        """Whether a CUDA device has the bitsandbytes its transformers needs for 4-bit layers; if not, it sits out 4-bit runs and
        the events say what to install."""
        libraries = self.workers[w].get("libraries") or {}
        if "transformers" not in libraries:
            return True
        have, need = libraries.get("bitsandbytes"), libraries.get("bitsandbytes-needed")
        if have and (not need or _version_tuple(have) >= _version_tuple(need)):
            return True
        key = (self.run_id, w)
        if key not in self.told_bnb and not getattr(self._planning, "previewing", False):
            self.told_bnb.add(key)
            want = f"bitsandbytes>={need}" if need else "bitsandbytes"
            self.event(f"{self.workers[w]['name']} sits out this 4-bit run: it needs {want} (it has {have or 'none'}); "
                       f"run pip install -U \"{want}\" on it and join again")
        return False

    def _cost(self, stages) -> float:
        """Seconds a step takes with this split: the pieces of a batch follow each other through the devices, so the first one
        goes through every stage and each further one adds the slowest stage. With many pieces that is the slowest stage; with
        one, the sum of all of them."""
        secs = [st.sec for st in stages]
        pieces = self._job().get("pieces") or self._job().get("micro", 8)
        return sum(secs) + (max(1, pieces) - 1) * max(secs)

    def _stages(self, order: list[str]):
        return plan(self._nodes(order), self._job()["layers"], self._job().get("layer_gb", 0.0)) if order else None

    def _best_stages(self, alive: list[str]):
        """The fastest split, and it leaves out devices that only slow the run down when the others can hold the model without
        them. The model's ends run on the training machine, so devices differ only by speed and memory: the order tried is the
        joining order (devices on one network stay neighbours) and the roomiest-first one."""
        placed = lambda w: self.workers[w]["settings"].get("order")
        pinned = lambda w: layer_bounds(self.workers[w]["settings"].get("layers"), self._job()["layers"])[0] > 0 and not self.workers[w].get("oom_capped")

        def best_for(group):
            found = None
            nodes = {n.name: n for n in self._nodes(group)}
            spare = lambda w: nodes[w].cap_gb - nodes[w].fixed_gb if w in nodes else -1e9
            if any(placed(w) is not None for w in group):
                orders = [sorted(group, key=lambda w: (placed(w) is None, placed(w) or 0, group.index(w)))]
            else:
                orders = [list(group)]
                roomy = sorted(group, key=spare, reverse=True)
                if roomy != orders[0]:
                    orders.append(roomy)
            for order in orders:
                stages = self._stages(order)
                if stages:
                    cost = self._cost(stages)
                    if found is None or cost < found[0] - 1e-9:
                        found = (cost, stages)
            return found
        group = list(alive)
        best = best_for(group)
        while best is None and len(group) > 1:
            trials = [(best_for([w for w in group if w != out]), out) for out in group if not pinned(out)]
            feasible = [(t, out) for t, out in trials if t]
            if feasible:
                best, out = min(feasible, key=lambda x: x[0][0])
                group.remove(out)
                break
            nodes = {n.name: n for n in self._nodes(group)}
            loose = [w for w in group if not pinned(w)]
            if not loose:
                break
            group.remove(min(loose, key=lambda w: nodes[w].cap_gb - nodes[w].fixed_gb if w in nodes else -1e9))
            best = best_for(group)
        while best and len(group) > 1:
            trials = [(best_for([w for w in group if w != out]), out) for out in group if not pinned(out)]
            trials = [(t, out) for t, out in trials if t]
            if not trials:
                break
            (cost, stages), out = min(trials, key=lambda x: x[0][0])
            if cost >= 0.9 * best[0]:
                break
            group.remove(out)
            best = (cost, stages)
        return best[1] if best else None

    def _engine(self, info: dict) -> str:
        """What the device trains this run with: "mlx", "mps" (a Mac with PyTorch, for models MLX does not run), "cuda" or "cpu"."""
        kind = info["devices"][0]["kind"]
        if kind == "mlx" and not (self._job() and self._job().get("mlx_ok") and info.get("backend") == "mlx"
                                  and info.get("no_mlx") != self._job().get("repo")):
            return "mps"
        return kind

    def _guess(self, w: str) -> float:
        info = self.workers[w]
        chosen = info["settings"].get("gpus") if info["devices"][0]["kind"] == "cuda" else None
        return DEFAULT_SPEED.get(self._engine(info), 4.0) / len(chosen or info["devices"])

    def _calibration(self, alive: list[str]) -> float:
        ratios = sorted(self.measured[w] / self._guess(w) for w in alive if w in self.measured)
        return ratios[len(ratios) // 2] if ratios else 1.0

    def _measure(self, reports: dict[str, dict]):
        for w, r in reports.items():
            cores = r.get("cores")
            if not cores or not sum(c[0] for c in cores):
                continue
            fast = [c for c in cores if len(c) < 3 or c[2] != "spill"]
            spill = [c for c in cores if len(c) > 2 and c[2] == "spill"]
            for found, store in ((fast, self.measured), (spill, self.measured_spill)):
                if found and sum(c[0] for c in found):
                    per_layer = max(c[1] for c in found) / sum(c[0] for c in found)
                    store[w] = per_layer if w not in store else 0.5 * store[w] + 0.5 * per_layer
            send = float(r.get("send") or 0.0)
            self.measured_link[w] = send if w not in self.measured_link else 0.5 * self.measured_link[w] + 0.5 * send

    def preview(self, overrides: dict[str, dict] | None = None) -> list[dict] | None:
        """The split the devices would get with these settings changed ({device id: {setting: value}}), or None if none fits."""
        stages, _ = self._try(overrides)
        return [{"w": st.name, "name": self.workers[st.name]["name"], "range": [st.first, st.last], "gb": st.gb, "sec": st.sec,
                 "on_gpu": st.on_gpu} for st in stages] if stages else None

    def plan_for(self, job: dict) -> dict:
        """How a run of this job would be split over the devices here now, or why it would not fit; changes nothing. While a run
        is going, it answers for the devices as they are, the running one included."""
        with self.cond:
            pending, self._planning.previewing, self._planning.job = self.pending, True, job
            try:
                alive = [w for w, v in self.workers.items() if v["alive"] and v["settings"].get("enabled", True)]
                stages = self._best_stages(alive) if alive else None
                self.pending = False
                why = None if stages else (self._why_no_plan() or "no device has joined yet")
            finally:
                self.pending, self._planning.previewing, self._planning.job = pending, False, None
            return {"layers": job["layers"], "layer_gb": job.get("layer_gb"), "why": why,
                    "plan": [{"w": st.name, "name": self.workers[st.name]["name"], "range": [st.first, st.last], "gb": st.gb, "sec": st.sec,
                              "on_gpu": st.on_gpu}
                             for st in stages] if stages else None}

    def _try(self, overrides: dict[str, dict] | None):
        """The split with some devices' settings changed, and why there is none, without changing anything or telling anyone."""
        with self.cond:
            if self.job is None:
                return None, None
            saved = {w: dict(self.workers[w]) for w in (overrides or {})}
            measured, links, spills, self._planning.previewing = dict(self.measured), dict(self.measured_link), dict(self.measured_spill), True
            try:
                for w, change in (overrides or {}).items():
                    info = self.workers[w]
                    change = clean_settings(change, sum(1 for d in info["devices"] if d["kind"] == "cuda"))
                    info["settings"] = {**info["settings"], **change}
                    if "cap_gb" in change:
                        info["cap_gb"] = change["cap_gb"]
                    if "cap_gb" in change:
                        info.pop("oom_capped", None)
                    if "gpus" in change:
                        self.measured.pop(w, None)
                        self.measured_link.pop(w, None)
                alive = [w for w, v in self.workers.items() if v["alive"] and v["settings"].get("enabled", True)]
                stages = self._best_stages(alive) if alive else None
                pending, self.pending = self.pending, False
                why = None if stages else self._why_no_plan()
                self.pending = pending
            finally:
                for w, before in saved.items():
                    self.workers[w].clear()
                    self.workers[w].update(before)
                self.measured.clear()
                self.measured.update(measured)
                self.measured_link.clear()
                self.measured_link.update(links)
                self.measured_spill.clear()
                self.measured_spill.update(spills)
                self._planning.previewing = False
            return stages, why

    def _maybe_rebalance(self):
        if not self.job.get("rebalance", True) or self.steps_in_plan < 3 or not self.plan:
            return
        if self.steps_in_plan > 3 and (self.steps_in_plan - 3) % (5 if len(self.workers) <= 8 else 20):
            return
        alive = [p["w"] for p in self.plan]
        if any(w not in self.measured for w in alive):
            return
        now = self._cost([Stage(p["w"], 0, 0, 0.0, self.measured[p["w"]] * (p["range"][1] - p["range"][0]) + self.measured_link.get(p["w"], 0.0))
                          for p in self.plan])
        everyone = [w for w, v in self.workers.items() if v["alive"] and v["settings"].get("enabled", True)]
        stages = self._best_stages(everyone)
        if not stages:
            return
        better = self._cost(stages)
        moved = [(st.name, st.first, st.last) for st in stages] != [(p["w"], *p["range"]) for p in self.plan]
        left = max(1, (self.job.get("expected_steps") or self.step + 100) - self.step)
        if moved and better < 0.87 * now and (now - better) * left > self.job.get("rebalance_cost_s", 60):
            self._replan(f"rebalancing on measured speed, compute per step {now:.1f}s -> {better:.1f}s", lost=False)

    def _replan(self, why: str, lost: bool = True):
        self.steps_in_plan = 0
        alive = [w for w, v in self.workers.items() if v["alive"] and v["settings"].get("enabled", True)]
        stages = self._best_stages(alive) if alive and self.job else None
        self.epoch += 1
        self.pending = False
        if lost or self.plan is None:
            self.restore = self.complete_step()
        else:
            self.restore = self.step - 1
        self.protect = self.restore
        if self.step > self.restore + 1:
            self.event(f"rolling back to the state after step {self.restore}; steps {self.restore + 1}-{self.step - 1} will be replayed")
        self.step = self.restore + 1
        self.seq_base = self.seq
        self.plan = [{"w": s.name, "range": [s.first, s.last], **({"on_gpu": s.on_gpu} if s.on_gpu is not None else {})} for s in stages] \
            if stages else None
        self.server.inbox.drop(lambda k: False)
        text = ", ".join(f"{self.workers[p['w']]['name']} {p['range']}" for p in self.plan) if self.plan else "no split fits the devices"
        idle = [self.workers[w]["name"] for w in (self._usable(alive) if self.job else []) if self.plan and w not in {p["w"] for p in self.plan}]
        if idle:
            text += f"; {', '.join(idle)} sit{'s' if len(idle) == 1 else ''} this run out because the others are faster without {'it' if len(idle) == 1 else 'them'}"
        self.event(f"plan {self.epoch} ({why}): {text}")
        self.emit("plan", {"epoch": self.epoch, "plan": self.plan})
        self.cond.notify_all()

    def _monitor(self):
        while not self.closed:
            time.sleep(1)
            with self.cond:
                for w, info in self.workers.items():
                    if info["alive"] and time.time() - info["seen"] > self.heartbeat_timeout:
                        self._lose(w, f"lost (silent for {time.time() - info['seen']:.0f}s)")
                        break

    def set_device(self, wid: str, **changes) -> dict:
        with self.cond:
            info = self.workers[wid]
            gpu_count = sum(1 for d in info["devices"] if d["kind"] == "cuda")
            changes = clean_settings(changes, gpu_count)
            shaping = any(k in changes for k in ("cap_gb", "enabled", "layers", "gpus", "gpu_caps", "order", "ram_cap"))
            if self.job is not None and shaping and info["alive"] and changes.get("enabled", True):
                stages, why = self._try({wid: changes})
                if not stages and why:
                    raise PoolError(f"not changed, since the run could not go on: {why}")
            moved = {k: v for k, v in changes.items() if info["settings"].get(k) != v}
            settings = {**info["settings"], **changes, "version": info["settings"].get("version", 0) + 1}
            if "cap_gb" in changes:
                info.pop("oom_capped", None)
            info["settings"] = settings
            for key in ("name", "cap_gb"):
                if key in changes:
                    info[key] = changes[key]
            replan = self.job is not None and shaping
            if "gpus" in changes:
                self.measured.pop(wid, None)
                self.measured_spill.pop(wid, None)
                self.measured_link.pop(wid, None)
            if moved:
                self.event(f"{info['name']} settings changed: " + ", ".join(f"{k} {'auto' if v is None else v}" for k, v in moved.items()))
            self._remember(info)
            if replan:
                self.pending = True
                self.cond.notify_all()
            return settings

    def _remember(self, info: dict):
        if not self.settings_file or not info.get("identity"):
            return
        self.saved_settings[info["identity"]] = {k: info["settings"].get(k) for k in DEVICE_SETTINGS if k in info["settings"]}
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file + ".tmp", "w", encoding="utf-8") as f:
                json.dump(self.saved_settings, f, indent=1)
            replace_file(self.settings_file + ".tmp", self.settings_file)
        except OSError:
            pass

    def drain(self, wid: str):
        self.set_device(wid, enabled=False)

    def remove(self, wid: str):
        with self.cond:
            info = self.workers[wid]
            info["removed"] = True
            info["settings"]["enabled"] = False
            self._lose(wid, "was removed")
            if info.get("identity"):
                self.identities.pop(info["identity"], None)
            for t in [t for t, w in self.device_tokens.items() if w == wid]:
                del self.device_tokens[t]
            self._save_state()

    def devices(self) -> list[dict]:
        with self.cond:
            out = []
            for w, info in self.workers.items():
                stage = next((p for p in self.plan or [] if p["w"] == w), None)
                status = "removed" if info.get("removed") else "banned" if info.get("banned") else "offline" if not info["alive"] else \
                    "disabled" if not info["settings"].get("enabled", True) else "training" if stage and self.job else "idle"
                out.append({"id": w, "name": info["name"], "status": status, "backend": info.get("backend"), "devices": info["devices"],
                            "cap_gb": info.get("cap_gb"), "measured_s_per_layer": self.measured.get(w), "link_s": self.measured_link.get(w),
                            "layers": stage["range"] if stage else None, "on_gpu": stage.get("on_gpu") if stage else None, "last_seen_s": round(time.time() - info["seen"], 1),
                            "telemetry": info.get("telemetry", {}), "settings": info["settings"], "applied": info.get("applied", 0),
                            "near": info.get("near"), "version": info.get("version"),
                            "libraries": info.get("libraries", {}), "why_left_out": info.get("why_left_out") if info.get("banned") else None})
            return out

    def status(self) -> dict:
        with self.cond:
            return {"version": __version__, "run": self.run_id if self.job else None, "last_run": self.run_id, "job": self.job, "step": self.step, "epoch": self.epoch,
                    "plan": [{**p, "name": self.workers[p["w"]]["name"]} for p in self.plan or []] if self.job else None,
                    "devices": len([w for w in self.workers.values() if w["alive"]]), "history": _thin(self.history, 400), "steps_done": len(self.history),
                    "trained_s": round(sum(h.get("seconds") or 0.0 for h in self.history), 1),
                    "events": self.events[-50:], "runs": self.runs[-20:], "waiting": self.waiting()}

    def set_run(self, rebalance=None, resplit: bool = False, expected_steps: int | None = None) -> dict:
        """Change the running run: rebalance turns moving layers to faster devices on or off; resplit splits the layers again now
        (after changing devices' settings, or to undo moves); expected_steps is how long the run is, which rebalancing weighs
        moves against (the Trainer and the built-in loops tell it themselves)."""
        with self.cond:
            if self.job is None:
                raise PoolError("no run is going on; to start one without rebalancing, pass rebalance=False to it")
            if expected_steps:
                self.job["expected_steps"] = int(expected_steps)
            if rebalance is not None:
                self.job["rebalance"] = bool(rebalance)
                self.event(f"rebalancing turned {'on' if rebalance else 'off'}")
            if resplit:
                self.pending = True
                self.event("splitting the layers again, as asked")
                self.cond.notify_all()
            return {"rebalance": self.job.get("rebalance", True), "expected_steps": self.job.get("expected_steps")}

    def start_run(self, job: dict, run_id: str | None = None) -> str:
        with self.op_lock, self.cond:
            if self.job is not None:
                raise PoolError(f"run {self.run_id} is still active; end it first")
            if job.get("mode", "lora") == "full" and job.get("kind") == "q4":
                raise PoolError("full fine-tuning needs a model in 16/32-bit; a 4-bit model can only be trained with LoRA")
            if run_id is not None and not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", run_id) or run_id in (".", ".."):
                raise PoolError(f"run name {run_id!r}: use up to 64 letters, digits, '.', '_' or '-'")
            self.job = dict(job)
            self.run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S-") + secrets.token_hex(2)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.step, self.restore, self.protect = 0, -1, None
            self.ckpt = {}
            for name in os.listdir(self.ckpt_dir):
                if name.endswith((".npz", ".weights", ".part")):
                    try:
                        os.remove(os.path.join(self.ckpt_dir, name))
                    except OSError:
                        pass
            self._set_oplog([])
            shutil.rmtree(os.path.join(self.ckpt_dir, "replay"), ignore_errors=True)
            self.oplog_after = -1
            self._good_steps = 0
            self._scales = {}
            self._grad_sq = 0.0
            self.history = []
            self.measured = {}
            self.measured_spill = {}
            self.measured_link = {}
            self.accum = {"loss": None, "tokens": 0, "started": None}
            self.runs.append({"id": self.run_id, "started": time.time(), "job": self.job, "state": "running"})
            for info in self.workers.values():
                if info.get("banned") and not info.get("removed"):
                    info["banned"], info["fails"], info["alive"] = False, 0, time.time() - info["seen"] < self.heartbeat_timeout
            self.plan = None
            self._replan(f"run {self.run_id} started", lost=False)
            self._tell_slow_kernels([w for w, v in self.workers.items() if v["alive"]])
            return self.run_id

    def end_run(self, state: str = "finished", run_id: str | None = None):
        with self.cond:
            if self.job is None or (run_id is not None and run_id != self.run_id):
                return
            for r in self.runs:
                if r["id"] == self.run_id:
                    r["state"], r["ended"], r["steps"] = state, time.time(), self.step
            self.event(f"run {self.run_id} {state} after {self.step} steps")
            self.job = None
            self.plan = None
            self.epoch += 1
            self.cond.notify_all()

    def wait_ready(self, timeout: float | None = None):
        deadline = None if timeout is None else time.time() + timeout
        with self.cond:
            while True:
                if self.job is None:
                    raise PoolError("no active run")
                if self.pending:
                    self._replan("a device joined, came back or changed", lost=False)
                if self.plan and all((p["w"], self.epoch) in self.ready for p in self.plan):
                    self.protect = None
                    return
                if not self.plan:
                    why = self._why_no_plan()
                    if why:
                        raise PoolError(why)
                left = None if deadline is None else deadline - time.time()
                if left is not None and left <= 0:
                    raise TimeoutError("the devices did not get ready in time; check the dashboard or `layerpool devices`")
                self.cond.wait(timeout=1 if left is None else min(1, left))

    def _set_oplog(self, kept: list[dict]):
        """The replay log becomes `kept`; the files of the operations it drops go."""
        ids = {id(op) for op in kept}
        for op in self.oplog:
            if id(op) not in ids:
                for key in ("inputs", "cot"):
                    if isinstance(op.get(key), Spilled):
                        op[key].drop()
        self.oplog = kept

    def _spilled(self, op: dict, seq: int) -> dict:
        """An operation the devices did, as the replay log keeps it: its inputs and loss gradients in files."""
        logged = {k: v for k, v in op.items() if k != "provider"}
        for key in ("inputs", "cot"):
            if logged.get(key) is not None and not isinstance(logged[key], Spilled):
                folder = os.path.join(self.ckpt_dir, "replay")
                os.makedirs(folder, exist_ok=True)
                logged[key] = Spilled(os.path.join(folder, f"{self.run_id}_op{seq}_{key}.log"), logged[key])
        return logged

    def _oplog_gb(self) -> float:
        total = 0
        for op in self.oplog:
            for key in ("inputs", "cot"):
                blobs = op.get(key) or []
                total += blobs.nbytes if isinstance(blobs, Spilled) else sum(len(b) for b in blobs if isinstance(b, (bytes, bytearray)))
        return total / 1e9

    def _need_run(self):
        if self.job is None:
            raise PoolError("this run has ended (from the dashboard, or because its script stopped answering); start a new run")

    def _why_no_plan(self) -> str | None:
        """Why no split exists when waiting cannot change it, in words the user can act on; None while devices may still come."""
        known = [w for w, v in self.workers.items() if not v.get("removed")]
        alive = [w for w in known if self.workers[w]["alive"]]
        errors = [self.workers[w].get("why_left_out") for w in known if self.workers[w].get("banned")]
        if errors and not alive:
            return f"every device left the run after errors; the last one: {errors[-1]}"
        enabled = [w for w in alive if self.workers[w]["settings"].get("enabled", True)]
        if alive and not enabled:
            return "every device is paused: resume one in the dashboard or with `layerpool resume DEVICE`"
        usable = self._usable(enabled)
        if enabled and not usable:
            return ("no device here can run this 4-bit model: it needs a GPU bitsandbytes runs on (NVIDIA, or a recent AMD one), "
                    "or an Apple Silicon Mac with an MLX 4-bit version of the model (mlx_model=)")
        if usable and not self.pending:
            nodes = self._nodes(usable)
            n = self._job()["layers"]
            name = lambda node: self.workers[node.name]["name"]
            fixed = [node for node in nodes if node.min_layers]
            if sum(node.min_layers for node in fixed) > n:
                return (f"the layer settings ask for {sum(node.min_layers for node in fixed)} layers ("
                        + ", ".join(f"{name(node)} {node.min_layers}" for node in fixed) + f") but the model has {n}: lower them or "
                        "set some to auto")
            if all(node.max_layers is not None for node in nodes) and sum(node.max_layers for node in nodes) < n:
                return (f"the layer settings allow at most {sum(node.max_layers for node in nodes)} layers in all and the model has {n}: "
                        "raise them, set one to auto, or add a device")
            for node in fixed:
                need = node.fixed_gb + node.min_layers * (node.layer_gb or self._job().get("layer_gb", 0.0))
                if need > node.cap_gb:
                    return (f"{name(node)} is set to at least {node.min_layers} layers, which need about {need:.1f} GB, and it may use "
                            f"{node.cap_gb:.1f} GB: lower its layers or raise its memory cap")
            per_layer = max(((n.layer_gb or self._job().get("layer_gb", 0.0)) for n in nodes), default=0.0)
            need = per_layer * self._job()["layers"]
            room = ", ".join(f"{self.workers[n.name]['name']} {max(0.0, n.cap_gb - n.fixed_gb):.1f} GB (of {n.cap_gb:.1f} GB it may use, "
                             f"{n.fixed_gb:.1f} GB go to activations and other parts)" for n in nodes)
            return (f"the model does not fit the devices: its {self._job()['layers']} blocks need about {need:.1f} GB "
                    f"({per_layer:.2f} GB each) and the devices have room for {room}. Add devices, raise their memory caps, use "
                    "LoRA instead of full fine-tuning, a 4-bit model, or a shorter seq")
        return None

    def _broadcast(self, epoch: int, seq: int, stages: list[dict], header: dict):
        """Sends the operation to every device at once and returns a function that waits for that and raises what failed, so
        the first device's inputs go out at the same time instead of a round trip later."""
        data = json.dumps(header).encode()
        errors = []

        def one(w):
            try:
                self.workers[w]["peer"].send(("op", epoch, seq), data, cancel=lambda: self.epoch != epoch, retries=2)
            except Cancelled:
                pass
            except ConnectionError as e:
                errors.append(Unreachable(w, e))
        threads = [threading.Thread(target=one, args=(p["w"],), daemon=True) for p in stages]
        [t.start() for t in threads]

        def wait():
            [t.join() for t in threads]
            if errors:
                raise errors[0]
        return wait

    def _send(self, wid: str, key: tuple, data: bytes, epoch: int):
        try:
            self.workers[wid]["peer"].send(key, data, cancel=lambda: self.epoch != epoch, retries=2)
        except Cancelled:
            raise Interrupted()
        except ConnectionError as e:
            raise Unreachable(wid, e)

    def _recv(self, key: tuple, epoch: int) -> bytes:
        while True:
            data = self.server.inbox.get(tuple(str(x) for x in key), timeout=0.2)
            if data is not None:
                return data
            if self.epoch != epoch:
                raise Interrupted()

    def _wait_reports(self, epoch: int, seq: int, stages: list[dict]) -> dict[str, dict]:
        started, told = time.time(), 0.0
        with self.cond:
            while self.epoch == epoch and not all((p["w"], epoch, seq) in self.reports for p in stages):
                self.cond.wait(timeout=1)
                waited = time.time() - started
                usual = max([1.0] + [h.get("seconds") or 0.0 for h in self.history[-5:]])
                if waited > max(30.0, 4 * usual) and waited - told > 30.0:
                    told = waited
                    slow = [self.workers[p["w"]]["name"] for p in stages if (p["w"], epoch, seq) not in self.reports]
                    self.event(f"waiting {waited:.0f}s for {', '.join(slow)} (a step usually takes {usual:.0f}s); a device that stays "
                               f"silent for {self.heartbeat_timeout:.0f}s is dropped and its layers move")
            if self.epoch != epoch:
                raise Interrupted()
            return {p["w"]: self.reports.pop((p["w"], epoch, seq)) for p in stages}

    def _run_op(self, op: dict, epoch: int, seq: int, stages: list[dict]) -> dict:
        header = {k: v for k, v in op.items() if k not in ("inputs", "cot", "provider")}
        first, last = stages[0]["w"], stages[-1]["w"]
        outputs, grads = None, None
        if op["type"] in ("fb", "fwd"):
            n = len(op["inputs"])
            header["n"] = n
            header["outputs"] = op["type"] == "fwd" or op.get("cot") is None
            header["grads"] = op["type"] == "fb" and bool(op.get("grads"))
        sent = self._broadcast(epoch, seq, stages, header)
        if op["type"] in ("fb", "fwd"):
            for m, blob in enumerate(op["inputs"]):
                self._send(first, ("d", epoch, seq, m), blob, epoch)
        sent()
        if op["type"] in ("fb", "fwd"):
            if header["outputs"]:
                outputs = [self._recv(("o", epoch, seq, m), epoch) for m in range(n)]
            if op["type"] == "fb":
                if op.get("cot") is None:
                    op["cot"] = list(op["provider"](outputs))
                for m, blob in enumerate(op["cot"]):
                    self._send(last, ("cg", epoch, seq, m), blob, epoch)
                if header["grads"]:
                    grads = [self._recv(("g0", epoch, seq, m), epoch) for m in range(n)]
        reports = self._wait_reports(epoch, seq, stages)
        return {"reports": reports, "outputs": outputs, "grads": grads, "last": last}

    def _recover(self):
        while True:
            self.wait_ready()
            with self.cond:
                epoch, stages = self.epoch, [dict(p) for p in self.plan]
                if self.restore < self.oplog_after:
                    lost = f"the saved state of step {self.oplog_after} was lost together with the device that held it"
                    self.event(f"run {self.run_id} cannot continue exactly: {lost}. Start the run again from your last save.")
                    raise PoolError(f"cannot recover the run exactly: {lost}; start again from your last save")
                replay = [{**op, "grads": False} for op in self.oplog if op["step"] > self.restore]
            try:
                for op in replay:
                    seq = self._next_seq()
                    self._run_op(op, epoch, seq, stages)
                    if op["type"] == "commit":
                        with self.cond:
                            self.step = op["step"] + 1
                with self.cond:
                    if self.epoch != epoch:
                        continue
                    self._replayed_epoch = epoch
                if replay:
                    self.event(f"replayed {len(replay)} operations, the run continues exactly where it was")
                return
            except (Interrupted, Unreachable) as e:
                self._handle(e, epoch)

    def _next_seq(self) -> int:
        with self.cond:
            seq = self.seq
            self.seq += 1
            return seq

    def _handle(self, error: Exception, epoch: int):
        if isinstance(error, Unreachable) and not self._answers(error.wid):
            with self.cond:
                if self.epoch == epoch:
                    self._lose(error.wid, "does not answer (the pool could not reach it)")
        with self.cond:
            if self.epoch == epoch:
                self._replan(f"an operation could not finish: {error!r}")

    def execute(self, op: dict, log: bool = True) -> dict:
        with self.op_lock:
            while True:
                if self.job is None:
                    raise PoolError("no active run")
                self.wait_ready()
                if self._replayed_epoch != self.epoch:
                    self._recover()
                    continue
                with self.cond:
                    epoch, stages = self.epoch, [dict(p) for p in self.plan]
                seq = self._next_seq()
                try:
                    if op["type"] == "commit":
                        op["step"] = self.step
                    elif op["type"] == "fb":
                        op["step"] = self.step
                    result = self._run_op(op, epoch, seq, stages)
                except (Interrupted, Unreachable) as e:
                    self._handle(e, epoch)
                    continue
                except BaseException as e:
                    with self.cond:
                        if self.job is not None and self.epoch == epoch:
                            self._replan(f"an operation was stopped ({type(e).__name__}); the devices drop it and keep the run as it was",
                                         lost=op["type"] == "commit")
                    raise
                if log and op["type"] in ("fb", "commit"):
                    logged = self._spilled(op, seq)
                    with self.cond:
                        self.oplog.append(logged)
                if op["type"] == "commit":
                    with self.cond:
                        self.step = op["step"] + 1
                        for p in stages:
                            self.workers[p["w"]]["fails"] = 0
                return result

    def fit_batch(self, rows: int, seq: int, pieces: int | None = None):
        """The training code's batches have pieces of `rows` rows of `seq` tokens. When that is more than the run assumed, the
        memory each device needs is worked out again, and the split changes if it no longer fits."""
        from .job import estimate, hf_config
        with self.cond:
            job = self.job
            if job is None:
                return {}
            have_rows, have_seq = job.get("piece_rows") or job.get("micro_batch_size") or 1, job.get("seq", 512)
            if pieces and int(pieces) != job.get("pieces"):
                job["pieces"] = max(1, int(pieces))
            if rows <= have_rows and seq <= have_seq:
                return {}
            rows, seq = max(int(rows), have_rows), max(int(seq), have_seq)
            run_id = self.run_id
        sizes = dict(job.get("sizes") or {})
        try:
            configs = self.__dict__.setdefault("_configs", {})
            config = configs.get(job["repo"]) or configs.setdefault(job["repo"], hf_config(job["repo"]))
            sizes["torch"] = estimate(job["repo"], job["layout"], job["rank"], seq, rows, job["micro"], job["mode"], config)
        except Exception as e:
            self.event(f"could not work out the memory of {rows}-row pieces of {seq} tokens: {e!r}")
            return {}
        if "mlx" in sizes and job.get("mlx_repo"):
            try:
                sizes["mlx"] = estimate(job["mlx_repo"], job["layout"], job["rank"], seq, rows, job["micro"], job["mode"], config, stack="model.layers")
            except Exception:
                pass
        with self.cond:
            if self.job is not job or self.run_id != run_id:
                return {}
            job.update(sizes["torch"])
            job["sizes"], job["seq"], job["piece_rows"] = sizes, seq, rows
            alive = [w for w, v in self.workers.items() if v["alive"] and v["settings"].get("enabled", True)]
            stages = self._best_stages(alive) if alive else None
            now = [(s.name, s.first, s.last) for s in stages] if stages else None
            if now != [(p["w"], *p["range"]) for p in self.plan or []]:
                self._replan(f"pieces of {rows} rows of {seq} tokens need more memory than the run assumed", lost=False)
            return {"rows": rows, "seq": seq}

    def forward_backward(self, inputs: list[bytes], provider, tokens: int = 0, grads: bool = False) -> dict:
        """One batch through the devices and back. `inputs` are the packed stack inputs of each micro-batch; `provider(outputs)` gets
        the packed stack outputs and returns the packed output gradients. Returns the gradients of the stack inputs when `grads`."""
        op = {"type": "fb", "inputs": inputs, "cot": None, "provider": provider, "grads": grads}
        t = time.time()
        if self.accum["started"] is None:
            self.accum["started"] = t
        result = self.execute(op)
        reports = result["reports"]
        with self.cond:
            if self.steps_in_plan > 0:
                self._measure(reports)
            self.accum["tokens"] += tokens
            self._last_reports = reports
            self._tell_scales(reports)
            self._grad_sq = sum(r.get("grad_sq", 0.0) for r in reports.values())
        return {"grads": result["grads"], "seconds": round(time.time() - t, 3), "grad_sq": self._grad_sq,
                "timing": {self.workers[w]["name"]: {"compute": max([c[1] for c in r.get("cores", [])] or [r.get("compute", 0)]), "wait": r.get("wait", 0),
                                                     "send": r.get("send", 0)} for w, r in reports.items()}}

    def _device_scales(self) -> list[float]:
        return [r["loss_scale"] for r in (getattr(self, "_last_reports", None) or {}).values() if isinstance(r.get("loss_scale"), (int, float))]

    def _tell_scales(self, reports: dict):
        """Devices lower their own loss scale when a 16-bit backward overflows and redo that piece; say so once per new low."""
        seen = self.__dict__.setdefault("_scales", {})
        for w, r in reports.items():
            scale = r.get("loss_scale")
            if isinstance(scale, (int, float)) and w in seen and scale < seen[w]:
                self.event(f"{self.workers[w]['name']} lowered its loss scale to {scale:g}: a float16 backward overflowed there, and "
                           "it redid that piece instead of skipping the step")
            if isinstance(scale, (int, float)):
                seen[w] = scale

    def add_loss(self, value: float):
        with self.cond:
            self.accum["loss"] = (self.accum["loss"] or 0.0) + float(value)

    def forward(self, inputs: list[bytes], adapter: bool = True) -> list[bytes]:
        return self.execute({"type": "fwd", "inputs": inputs, "adapter": adapter}, log=False)["outputs"]

    def optim_step(self, lr: float, max_grad_norm: float | None = None, loss: float | None = None, clip: float | None = None,
                   weight_decay: float | None = None, skip: bool = False) -> dict:
        """One optimizer step on every device. skip=True drops the gradients gathered since the last step instead, as the training
        code did (a GradScaler that found an overflow in its own part)."""
        self._need_run()
        norm = float(np.sqrt(getattr(self, "_grad_sq", 0.0)))
        overflow = not np.isfinite(norm) or (clip is not None and not np.isfinite(clip))
        if skip and not overflow:
            clip = 0.0
            self.event(f"step {self.step}: the training code dropped this step's gradients without a step, so the devices drop theirs too")
        skip = skip or overflow
        if overflow:
            clip = 0.0
            scale = max(1.0, min([self.job.get("loss_scale", 1.0)] + self._device_scales()) / 2)
            self.event(f"step {self.step} skipped: the gradients overflowed; loss scale lowered to {scale:g}")
            if self.job.get("loss_scale", 1.0) <= 1.0:
                halves = [self.workers[p["w"]]["name"] for p in self.plan or [] if self.workers[p["w"]]["devices"][0]["kind"] == "cuda"]
                self.event("the gradients overflow even without loss scaling: the values of this model do not fit in float16, which "
                           f"older GPUs train in ({', '.join(halves) or 'some devices'}). Use GPUs with bfloat16 (NVIDIA RTX 30 / A100 "
                           "and newer, AMD MI200 / RX 7000 and newer, Apple Silicon) or pause the float16 ones")
            self.job["loss_scale"], self._good_steps = scale, 0
        elif clip is None:
            clip = min(1.0, max_grad_norm / (norm + 1e-6)) if max_grad_norm else 1.0
        good = 0 if skip else self._good_steps + 1
        op = {"type": "commit", "lr": lr, "clip": clip, "skip": skip}
        if overflow:
            op["lower_scale"] = self.job["loss_scale"]
        if self._oplog_gb() > OPLOG_LIMIT_GB:
            op["checkpoint"] = True
        if weight_decay is not None:
            op["weight_decay"] = float(weight_decay)
        self.execute(op)
        self._good_steps = good
        with self.cond:
            self._grad_sq = 0.0
            took = time.time() - (self.accum["started"] or time.time())
            value = self.accum["loss"] if loss is None else loss
            record = {"step": self.step - 1, "loss": value, "lr": lr, "grad_norm": round(norm, 5) if np.isfinite(norm) else None, "clip": round(clip, 5),
                      "skipped": skip, "seconds": round(took, 2),
                      "tokens": self.accum["tokens"], "tokens_per_s": round(self.accum["tokens"] / took, 1) if took > 0 else None,
                      "plan": [[self.workers[p["w"]]["name"], p["range"]] for p in self.plan or []], "time": time.time()}
            if getattr(self, "_last_reports", None):
                record["timing"] = {self.workers[w]["name"]: {"compute": max([c[1] for c in r.get("cores", [])] or [r.get("compute", 0)]),
                                                              "wait": r.get("wait", 0), "send": r.get("send", 0)} for w, r in self._last_reports.items()}
            self.history.append(record)
            if len(self.history) > 20000:
                self.history = self.history[:-10000:2] + self.history[-10000:]
            self.accum = {"loss": None, "tokens": 0, "started": None}
            self.steps_in_plan += 1
            self._maybe_rebalance()
        said = "" if value is None else f" loss {value:.4f}"
        self.event(f"step {record['step']}{said} in {record['seconds']:.1f}s" + (f" ({record['tokens_per_s']} tok/s)" if record["tokens_per_s"] else ""))
        self.emit("step", record)
        return record

    def gather(self, step: int | None = None, moments: bool = False) -> dict[str, np.ndarray]:
        with self.cond:
            step = self.step - 1 if step is None else step
            units = self._units()
        out = {}
        for unit in units:
            path = self._path(unit, step)
            path = path + ".weights" if os.path.exists(path + ".weights") else path
            with open(path, "rb") as f:
                meta, arrays = unpack(f.read())
            arrays = from_half(meta, arrays)
            out.update({k: v for k, v in arrays.items() if moments or not k.endswith((".m", ".v", ".t"))})
        return out

    def load_weights(self, weights: dict[str, np.ndarray]):
        """Put saved weights (an adapter or trained blocks, under the devices' names such as "3.mlp.up_proj.A") on the devices.
        Before the first step they take the place of the initial values. Later (the Trainer's load_best_model_at_end, or going back
        to a save) they become the state after the last step taken, under a step number no checkpoint has used, so no older state
        of the devices can be taken for them; the run goes on from there, and a device that joins later starts from them too."""
        with self.op_lock, self.cond:
            if self.job is None:
                raise PoolError("no run is going")
            by_unit: dict[int, dict] = {}
            for key, value in weights.items():
                by_unit.setdefault(int(key.split(".", 1)[0]), {})[key] = np.asarray(value)
            lacking = [u for u in self._units() if u not in by_unit]
            if lacking:
                raise PoolError(f"the saved weights have nothing for blocks {lacking[:8]}")
            midway = self.step > 0
            at = self.step if midway else -1
            for unit, state in by_unit.items():
                path = self._path(unit, at)
                with open(path + ".part", "wb") as f:
                    f.write(pack(None, **state))
                replace_file(path + ".part", path)
                if midway:
                    self.ckpt[unit] = {}
                self._add(unit, at, "coord")
            if midway:
                self._set_oplog([])
                self.oplog_after, self.protect = at, at
                self._replan("loaded saved weights", lost=True)
            else:
                self.job["base"] = self.job.get("base", 0) + 1
                self._replan("loaded saved weights", lost=False)
            self.cond.notify_all()
        self.event(f"run {self.run_id} goes on from saved weights ({len(weights)} tensors)")

    def export(self, path: str, dtype: str = "bfloat16", metrics: list | None = None, optimizer: bool = False) -> str:
        """The run's result in path. optimizer=True also keeps the devices' AdamW state (layerpool_optimizer.safetensors), so a
        training checkpoint resumes exactly."""
        self._need_run()
        if optimizer:
            from safetensors.numpy import save_file
            self.execute({"type": "save", "target": "coord", "weights_only": True, "moments": True}, log=False)
            os.makedirs(path, exist_ok=True)
            full = self.gather(moments=True)
            save_file({k: np.ascontiguousarray(v) for k, v in full.items()}, os.path.join(path, OPTIMIZER_FILE))
            weights = {k: v for k, v in full.items() if not k.endswith((".m", ".v", ".t"))}
        else:
            half = dtype if self.job.get("mode", "lora") == "full" else None
            self.execute({"type": "save", "target": "coord", "weights_only": True, "dtype": half}, log=False)
            weights = self.gather()
        for name in os.listdir(self.ckpt_dir):
            if name.startswith(f"{self.run_id}_") and name.endswith(".weights"):
                os.remove(os.path.join(self.ckpt_dir, name))
        job = self.job
        if job.get("mode", "lora") == "lora":
            export_peft({k: v for k, v in weights.items() if k.endswith((".A", ".B"))}, path, job)
        else:
            export_blocks(weights, path, job, dtype)
        with open(os.path.join(path, "layerpool_run.json"), "w", encoding="utf-8") as f:
            json.dump({"run": self.run_id, "job": job, "steps": self.step, "history": self.history, "metrics": metrics or []}, f, indent=1, default=str)
        self.event(f"saved run {self.run_id} at step {self.step - 1} to {path}")
        return path

    def invite(self, ttl: float = 86400, reusable: bool = False) -> dict:
        ttl = float(ttl)
        if not 60 <= ttl <= 30 * 86400:
            raise ValueError("an invite lasts from a minute to 30 days")
        code = "inv_" + secrets.token_urlsafe(12)
        with self.cond:
            now = time.time()
            for old in [c for c, v in self.invites.items() if v["expires"] < now]:
                del self.invites[old]
            self.invites[code] = {"expires": time.time() + ttl, "reusable": reusable}
        url = self.urls[-1]
        join = f"layerpool join {url} --token {code}"
        return {"code": code, "url": url, "expires_in_s": ttl, "reusable": reusable, "tunnel": self.tunnel is not None,
                "command": f"pip install -U layerpool=={__version__}\n{join}",
                "notebook": f"!pip install -q -U layerpool=={__version__}\n!{join}"}

    def _check(self, handler, method: str, path: str, query: dict):
        query.pop("_invite", None)
        query.pop("_ip", None)
        token = handler.headers.get("X-Token")
        if token and hmac.compare_digest(token.encode("utf-8", "surrogateescape"), self.token.encode()):
            return None
        if token in self.device_tokens:
            if "w" in query and query["w"] != self.device_tokens[token]:
                return 403, {"error": "this device token belongs to another device"}
            if path != "/join" and self.device_tokens[token] not in self.workers:
                return 409, {"error": "rejoin"}
            return None
        if path == "/request":
            ip = handler.client_address[0]
            if self.tunnel is not None and ip in ("127.0.0.1", "::1"):
                ip = "".join(c for c in str(handler.headers.get("CF-Connecting-IP") or ip)[:64] if c.isprintable())
            query["_ip"] = ip
            if method == "POST":
                with self.cond:
                    now = time.time()
                    mine = [r for r in self.requests.values() if r.get("ip") == ip and r["status"] == "waiting" and now - r["time"] < self.REQUEST_TTL]
                    if len(mine) >= 3:
                        return 429, {"error": "this address already has devices waiting for approval; approve or deny them first"}
            return None
        if path == "/info":
            return None
        if path in ("/join", "/info") and token in self.invites:
            invite = self.invites[token]
            if invite["expires"] < time.time():
                return 403, {"error": "this invite has expired; create a new one from the dashboard or `layerpool invite`"}
            query["_invite"] = token
            return None
        return 403, b""

    def close(self, grace: float = 3.0):
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        deadline = time.time() + grace
        with self.cond:
            while time.time() < deadline and not all(not w["alive"] or w.get("bye") for w in self.workers.values()):
                self.cond.wait(timeout=0.1)
        time.sleep(0.2)
        if self.tunnel:
            self.tunnel.close()
        self.server.close()
        if self._own_ckpt:
            shutil.rmtree(self.ckpt_dir, ignore_errors=True)

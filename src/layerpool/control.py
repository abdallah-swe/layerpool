import hmac
import json
import os
import queue
import secrets
import threading
import time
import urllib.request

import numpy as np

from .codec import pack, replace_file, unpack
from .net import PeerServer
from .pool import PoolError

UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
STATE_FILE = os.path.join(os.path.expanduser("~"), ".layerpool", "pool.json")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
PAGE_HEADERS = {"Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
                                           "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
                "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}
LOGIN_TTL_S = 300


def _arrays(prefix: str, arrays: dict) -> list:
    out, i = [], 0
    while f"{prefix}{i}" in arrays:
        out.append(arrays[f"{prefix}{i}"])
        i += 1
    return out


def _blobs(prefix: str, arrays: dict) -> list[bytes]:
    return [a.tobytes() for a in _arrays(prefix, arrays)]


def _as_arrays(prefix: str, blobs) -> dict:
    return {f"{prefix}{i}": np.frombuffer(b, np.uint8) for i, b in enumerate(blobs or [])}


class EventStream:
    content_type = "text/event-stream"

    def __init__(self, pool):
        self.pool = pool
        self.q: queue.Queue = queue.Queue(maxsize=1000)
        self.listener = lambda kind, data: self._put(kind, data)
        pool.listeners.append(self.listener)

    def _put(self, kind, data):
        try:
            self.q.put_nowait((kind, data))
        except queue.Full:
            pass

    def __iter__(self):
        yield b"retry: 2000\n\n"
        while not self.pool.closed:
            try:
                kind, data = self.q.get(timeout=10)
                yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n".encode()
            except queue.Empty:
                yield b": keep-alive\n\n"

    def close(self):
        if self.listener in self.pool.listeners:
            self.pool.listeners.remove(self.listener)


class ControlServer:
    """Admin API and dashboard, on 127.0.0.1 only. Every dashboard button, CLI command and library call goes through these routes."""

    def __init__(self, pool, port: int = 0, token: str | None = None):
        self.pool = pool
        self.token = token or secrets.token_urlsafe(18)
        self.pending: dict[str, dict] = {}
        self.client_seen: dict[str, float] = {}
        self.logins: dict[str, float] = {}
        routes = {
            ("GET", "/"): self._index, ("GET", "/ui"): self._static, ("GET", "/api/status"): lambda q, b: (200, self._status()),
            ("GET", "/api/devices"): lambda q, b: (200, self.pool.devices()), ("GET", "/api/events"): lambda q, b: (200, EventStream(self.pool)),
            ("POST", "/api/device"): self._device, ("POST", "/api/device/drain"): lambda q, b: self._call(self.pool.drain, json.loads(b)["id"]),
            ("POST", "/api/device/remove"): lambda q, b: self._call(self.pool.remove, json.loads(b)["id"]),
            ("POST", "/api/invite"): lambda q, b: self._call(self.pool.invite, **(json.loads(b) if b else {})),
            ("GET", "/api/waiting"): lambda q, b: (200, self.pool.waiting()),
            ("POST", "/api/answer"): lambda q, b: self._call(self.pool.answer, **json.loads(b)),
            ("POST", "/api/preview"): lambda q, b: self._call(lambda: {"plan": self.pool.preview(json.loads(b) if b else None)}),
            ("POST", "/api/run/settings"): lambda q, b: self._call(self.pool.set_run, **json.loads(b)),
            ("POST", "/api/plan"): lambda q, b: self._call(self.pool.plan_for, json.loads(b)["job"]),
            ("POST", "/api/run/start"): self._start, ("POST", "/api/run/end"): self._end,
            ("POST", "/api/run/alive"): self._alive, ("POST", "/api/login"): self._login,
            ("POST", "/api/run/ready"): lambda q, b: self._call(self.pool.wait_ready, float(q.get("timeout", 1800))),
            ("POST", "/api/run/fb"): self._fb, ("POST", "/api/run/cot"): self._cot, ("POST", "/api/run/forward"): self._forward, ("POST", "/api/run/load"): self._load,
            ("POST", "/api/run/step"): lambda q, b: self._call(self.pool.optim_step, **json.loads(b)),
            ("POST", "/api/run/fit"): lambda q, b: self._call(self.pool.fit_batch, **json.loads(b)),
            ("POST", "/api/run/export"): lambda q, b: self._call(self.pool.export, **json.loads(b)),
            ("POST", "/api/run/metric"): lambda q, b: (self.pool.emit("metric", json.loads(b)), (200, b""))[1],
        }
        self.server = PeerServer(port=port, token=self.token, routes=routes, host="127.0.0.1", check=self._check)
        self.port = self.server.port
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self._watch_clients, daemon=True).start()

    def _status(self) -> dict:
        out = self.pool.status()
        seen = self.client_seen.get(out.get("run")) if out.get("run") else None
        out["script_quiet_s"] = round(time.monotonic() - seen, 1) if seen else None
        out["abandon_after_s"] = self.ABANDONED_AFTER_S
        return out

    def _watch_clients(self):
        """A training script that died (killed, crashed, out of memory) stops saying it is alive: end its run so the devices are
        free for the next one."""
        last = time.monotonic()
        while not self.pool.closed:
            time.sleep(5)
            now = time.monotonic()
            if now - last > 15:
                self.client_seen = {k: max(v, now - 5) for k, v in self.client_seen.items()}
            last = now
            try:
                self._abandon_stale_run()
            except Exception:
                pass

    def _check(self, handler, method: str, path: str, query: dict):
        host = handler.headers.get("Host", "")
        if host not in (f"127.0.0.1:{self.port}", f"localhost:{self.port}"):
            return 403, {"error": "unexpected Host header"}
        origin = handler.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}"):
            return 403, {"error": "cross-site request refused"}
        if path == "/" or path.startswith("/ui") or path == "/api/login":
            return None
        auth = handler.headers.get("Authorization", "")
        given = auth[7:] if auth.startswith("Bearer ") else handler.headers.get("X-Token") or query.get("t")
        if not hmac.compare_digest(str(given or "").encode("utf-8", "surrogateescape"), self.token.encode()):
            return 401, {"error": "missing or wrong admin token"}
        if path in ("/api/run/fb", "/api/run/cot", "/api/run/forward", "/api/run/step", "/api/run/export", "/api/run/ready", "/api/run/metric"):
            self._touch()
        return None

    def _index(self, query, body):
        with open(os.path.join(UI_DIR, "index.html"), "rb") as f:
            return 200, f.read(), TYPES[".html"], PAGE_HEADERS

    def _static(self, query, body):
        name = os.path.basename(query.get("f", ""))
        path = os.path.join(UI_DIR, name)
        if not name or not os.path.isfile(path):
            return 404, b""
        with open(path, "rb") as f:
            return 200, f.read(), TYPES.get(os.path.splitext(name)[1], "application/octet-stream"), PAGE_HEADERS

    def _call(self, fn, *args, **kwargs):
        try:
            out = fn(*args, **kwargs)
        except PoolError as e:
            return 409, {"error": str(e)}
        except (ValueError, KeyError, TimeoutError, TypeError) as e:
            return 400, {"error": str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)}
        return 200, out if isinstance(out, (dict, list)) else {"result": out}

    def _device(self, query, body):
        data = json.loads(body)
        wid = data.pop("id")
        return self._call(self.pool.set_device, wid, **data)

    ABANDONED_AFTER_S = float(os.environ.get("LAYERPOOL_ABANDON_S", "120"))

    def _alive(self, query, body):
        run_id = json.loads(body).get("run") if body else None
        if run_id:
            self.client_seen[run_id] = time.monotonic()
        active = self.pool.job is not None and self.pool.run_id == run_id
        return 200, {"active": active}

    def _fail_pending(self, why: str):
        for slot in list(self.pending.values()):
            slot["cot"].put(PoolError(why))

    def _end(self, query, body):
        data = json.loads(body) if body else {}
        if self.pool.job is not None and data.get("run") in (None, self.pool.run_id):
            self._fail_pending("the run was ended")
        return self._call(self.pool.end_run, data.get("state", "finished"), data.get("run"))

    def _abandon_stale_run(self):
        run_id = self.pool.run_id if self.pool.job is not None else None
        if run_id is None:
            return
        seen = self.client_seen.get(run_id)
        if seen is None or time.monotonic() - seen < self.ABANDONED_AFTER_S:
            return
        self._fail_pending("the training script that owned this run stopped")
        self.pool.event(f"run {run_id} has no live training script any more; ending it so a new run can start")
        self.pool.end_run("abandoned", run_id)

    def _start(self, query, body):
        self._abandon_stale_run()
        data = json.loads(body)
        if not isinstance(data.get("job"), dict):
            return 400, {"error": "start a run with a job (see layerpool.job.build_job)"}
        return self._call(lambda: {"run": self.pool.start_run(data["job"], data.get("name"))})

    def _touch(self):
        if self.pool.job is not None:
            self.client_seen[self.pool.run_id] = time.monotonic()

    def _fb(self, query, body):
        meta, arrays = unpack(body)
        inputs = _blobs("in", arrays)
        slot = {"news": queue.Queue(), "cot": queue.Queue()}
        op_id = secrets.token_hex(8)
        self.pending[op_id] = slot

        def provider(outputs):
            slot["news"].put(("outputs", outputs))
            try:
                got = slot["cot"].get(timeout=1800)
            except queue.Empty:
                raise PoolError("the training script did not send its loss gradient within 30 minutes") from None
            if isinstance(got, Exception):
                raise got
            return got

        def work():
            try:
                out = self.pool.forward_backward(inputs, provider, tokens=int(meta.get("tokens", 0)), grads=bool(meta.get("grads")))
                slot["news"].put(("result", out))
            except Exception as e:
                slot["news"].put(("result", e))
        threading.Thread(target=work, daemon=True).start()
        return self._answer(op_id)

    def _answer(self, op_id: str):
        slot = self.pending[op_id]
        kind, out = slot["news"].get()
        if kind == "outputs":
            return 200, pack({"done": False, "op": op_id, "n": len(out)}, **_as_arrays("out", out))
        del self.pending[op_id]
        if isinstance(out, Exception):
            return (409 if isinstance(out, PoolError) else 500), {"error": str(out) if isinstance(out, PoolError) else repr(out)}
        grads = out.pop("grads", None)
        return 200, pack({"done": True, **out}, **_as_arrays("g", grads))

    def _cot(self, query, body):
        meta, arrays = unpack(body)
        slot = self.pending.get(meta["op"])
        if slot is None:
            return 404, {"error": "unknown operation"}
        slot["cot"].put(_blobs("cot", arrays))
        return self._answer(meta["op"])

    def _load(self, query, body):
        meta, arrays = unpack(body)
        return self._call(self.pool.load_weights, arrays)

    def _forward(self, query, body):
        meta, arrays = unpack(body)
        try:
            outputs = self.pool.forward(_blobs("in", arrays), adapter=bool(meta.get("adapter", True)))
        except PoolError as e:
            return 409, {"error": str(e)}
        return 200, pack({"n": len(outputs)}, **_as_arrays("out", outputs))

    def publish(self, path: str = STATE_FILE):
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        tmp = path + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as f:
            json.dump({"url": self.url, "token": self.token, "pid": os.getpid(), "workers": self.pool.urls, "started": time.time()}, f)
        replace_file(tmp, path)
        self.state_file = path

    def dashboard_url(self) -> str:
        return f"{self.url}/#t={self.token}"

    def login_url(self) -> str:
        """The dashboard with a code that works once, for a few minutes, so the admin token itself is not on the command line
        that opens the browser."""
        code = secrets.token_urlsafe(16)
        now = time.monotonic()
        self.logins = {c: t for c, t in self.logins.items() if now - t < LOGIN_TTL_S}
        self.logins[code] = now
        return f"{self.url}/#o={code}"

    def _login(self, query, body):
        code = str((json.loads(body) if body else {}).get("code", ""))
        made = self.logins.pop(code, None)
        if made is None or time.monotonic() - made > LOGIN_TTL_S:
            return 403, {"error": "this sign-in link was used already or is too old; open the dashboard link `layerpool up` printed"}
        return 200, {"token": self.token}

    def close(self):
        if getattr(self, "state_file", None):
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    if json.load(f).get("pid") == os.getpid():
                        os.remove(self.state_file)
            except (OSError, ValueError):
                pass
        self.server.close()


_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def find_local(path: str = STATE_FILE) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        import psutil
        if not psutil.pid_exists(info["pid"]):
            return None
        _LOCAL.open(info["url"] + "/ping", timeout=2).read()
        return info
    except (OSError, ValueError, KeyError):
        return None
    except Exception:
        return None


class RemoteError(RuntimeError):
    pass


class RemotePool:
    """The same methods as Pool, over the control API of a pool running in another process (`layerpool up`)."""

    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token
        self.history: list = []

    def _req(self, path: str, data: bytes | None = None, timeout: float = 3600) -> bytes:
        req = urllib.request.Request(self.url + path, data=data, method="POST" if data is not None else "GET",
                                     headers={"Authorization": f"Bearer {self.token}"})
        try:
            with _LOCAL.open(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except ValueError:
                pass
            if e.code == 409:
                raise PoolError(detail) from None
            raise RemoteError(f"{e.code}: {detail}") from None

    def _json(self, path: str, payload=None):
        out = self._req(path, None if payload is None else json.dumps(payload, default=str).encode())
        return json.loads(out) if out else None

    def status(self):
        status = self._json("/api/status")
        self.history = status.get("history", [])
        return status

    def devices(self):
        return self._json("/api/devices")

    def invite(self, **kw):
        return self._json("/api/invite", kw)

    def waiting(self):
        return self._json("/api/waiting")

    def answer(self, code: str, approve: bool = True) -> str:
        return self._json("/api/answer", {"code": code, "approve": approve})["result"]

    def set_device(self, wid: str, **changes):
        return self._json("/api/device", {"id": wid, **changes})

    def drain(self, wid: str):
        return self._json("/api/device/drain", {"id": wid})

    def set_run(self, **changes):
        return self._json("/api/run/settings", changes)

    def plan_for(self, job: dict) -> dict:
        return self._json("/api/plan", {"job": job})

    def remove(self, wid: str):
        return self._json("/api/device/remove", {"id": wid})

    def preview(self, overrides=None):
        return self._json("/api/preview", overrides or {})["plan"]

    def start_run(self, job: dict, name: str | None = None) -> str:
        run_id = self._json("/api/run/start", {"job": job, "name": name})["run"]
        threading.Thread(target=self._keep_alive, args=(run_id,), daemon=True).start()
        return run_id

    def _keep_alive(self, run_id: str):
        while True:
            try:
                if not self._json("/api/run/alive", {"run": run_id}).get("active"):
                    return
            except Exception:
                pass
            time.sleep(5)

    def end_run(self, state: str = "finished", run_id: str | None = None):
        return self._json("/api/run/end", {"state": state, "run": run_id})

    def wait_ready(self, timeout: float | None = None):
        return self._json(f"/api/run/ready?timeout={timeout or 1800}", {})

    def forward_backward(self, inputs, provider, tokens: int = 0, grads: bool = False) -> dict:
        meta, got = unpack(self._req("/api/run/fb", pack({"tokens": tokens, "grads": grads}, **_as_arrays("in", inputs))))
        while not meta.get("done"):
            cots = provider(_blobs("out", got))
            meta, got = unpack(self._req("/api/run/cot", pack({"op": meta["op"]}, **_as_arrays("cot", cots))))
        meta.pop("done", None)
        meta["grads"] = _blobs("g", got) if grads else None
        return meta

    def add_loss(self, value: float):
        pass

    def fit_batch(self, rows: int, seq: int, pieces: int | None = None):
        return self._json("/api/run/fit", {"rows": rows, "seq": seq, "pieces": pieces})

    def forward(self, inputs, adapter: bool = True):
        meta, got = unpack(self._req("/api/run/forward", pack({"adapter": adapter}, **_as_arrays("in", inputs))))
        return _blobs("out", got)

    def optim_step(self, lr: float, max_grad_norm=None, loss=None, clip=None, weight_decay=None, skip: bool = False) -> dict:
        record = self._json("/api/run/step", {"lr": lr, "max_grad_norm": max_grad_norm, "loss": loss, "clip": clip, "weight_decay": weight_decay,
                                              "skip": skip})
        self.history.append(record)
        return record

    def export(self, path: str, dtype: str = "bfloat16", metrics=None, optimizer: bool = False) -> str:
        return self._json("/api/run/export", {"path": path, "dtype": dtype, "metrics": metrics, "optimizer": optimizer})["result"]

    def load_weights(self, weights: dict):
        self._req("/api/run/load", pack(None, **weights))

    def emit(self, kind: str, data: dict):
        if kind == "metric":
            try:
                self._json("/api/run/metric", data)
            except Exception:
                pass

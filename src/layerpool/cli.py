import argparse
import json
import os
import re
import shutil
import signal
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

import numpy as np
import psutil

from . import __version__, worker as worker_module
from .api import connect, start
from .control import ControlServer, RemoteError, find_local
from .pool import Pool, PoolError

REFERENCE_FILE = "reference_logprobs.npy"
TRAIN_KEYS = ("model", "mlx_model", "data", "method", "mode", "rank", "alpha", "lr", "schedule", "warmup", "min_lr_ratio", "weight_decay", "steps",
              "epochs", "batch_size", "seq", "micro", "micro_batch_size", "codec", "ckpt_every", "max_grad_norm", "beta", "out", "random_init", "train_on", "name",
              "save_every", "resume", "target_modules")


def plain_error(e: BaseException) -> str:
    """A sentence for the errors people meet most, instead of a traceback."""
    name = type(e).__name__
    if name in ("RepositoryNotFoundError", "GatedRepoError"):
        return ("Hugging Face cannot find that model, or it needs you to log in and accept its terms: check the name, run "
                "`hf auth login` (or set HF_TOKEN) here and on every device")
    if name in ("URLError", "ConnectionRefusedError", "ConnectionResetError", "RemoteDisconnected"):
        reason = getattr(e, "reason", None) or e
        return f"cannot connect ({reason}): if this is the pool, is `layerpool up` still running? (LAYERPOOL_URL picks another pool)"
    if isinstance(e, (FileNotFoundError, ValueError, TypeError, RuntimeError, OSError)):
        return str(e) or name
    raise e


def _client():
    try:
        return connect(start_if_missing=False)
    except Exception as e:
        sys.exit(f"layerpool: {e}")


def up(argv):
    p = argparse.ArgumentParser(prog="layerpool up", description="Start a pool: devices join it once, runs come and go. Opens the dashboard.")
    p.add_argument("--port", type=int, default=7340, help="dashboard and control API, on 127.0.0.1 only")
    p.add_argument("--worker-port", type=int, default=7341, help="where devices connect")
    p.add_argument("--tunnel", action="store_true", help="let devices outside this network join through a cloudflared tunnel")
    p.add_argument("--no-work", action="store_true", help="this machine only coordinates and does not train")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--heartbeat", type=float, default=120.0, help="seconds of silence before a device counts as gone")
    p.add_argument("--advertise", help="this machine's address for other devices, such as 192.168.1.20, when the one it finds is "
                                       "wrong (inside Docker, WSL or a VPN); the pool and this machine's own device use it with their ports")
    worker_module.add_worker_args(p)
    args = p.parse_args(argv)
    found = find_local()
    if found:
        sys.exit(f"layerpool: a pool is already running here: {found['url']} (stop it first)")
    pool = Pool(port=_free(args.worker_port), heartbeat_timeout=args.heartbeat, tunnel=args.tunnel, advertise=args.advertise,
                settings_file=os.path.join(os.path.expanduser("~"), ".layerpool", "devices.json"),
                state_file=os.path.join(os.path.expanduser("~"), ".layerpool", "pool_state.json"))
    control = ControlServer(pool, port=_free(args.port, "127.0.0.1"))
    control.publish()
    awake = None if args.allow_sleep else worker_module.keep_awake()
    if not args.no_work:
        local = worker_module.Worker(f"http://127.0.0.1:{pool.server.port}", pool.token, args.name, args.cap_gb, tunnel=args.tunnel,
                                     backend=args.backend, cpu=args.cpu,
                                     identity=worker_module.device_identity(os.path.join(os.path.expanduser("~"), ".layerpool", "local_device_id")),
                                     log=lambda *a: print("[this device]", *a, flush=True), layers=args.layers, gpus=args.gpus,
                                     gpu_caps=args.gpu_cap, order=args.order, allow_sleep=args.allow_sleep, ram_cap=args.ram_cap,
                                     advertise=[args.advertise] if args.advertise else None)
        threading.Thread(target=local.run, daemon=True).start()
    invite = pool.invite(reusable=True, ttl=7 * 86400)
    print(f"layerpool {__version__}")
    print(f"  dashboard   {control.dashboard_url()}")
    print(f"  devices     {' or '.join(pool.urls)}")
    print("  add one     " + invite["command"].replace("\n", "\n              "))
    print("  your code   import layerpool; pool = layerpool.connect()", flush=True)
    threading.Thread(target=_tell_if_outdated, daemon=True).start()
    if not args.no_browser:
        webbrowser.open(control.login_url())

    def interrupt(*_):
        raise KeyboardInterrupt
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nlayerpool: stopping")
    finally:
        if awake is not None:
            awake.terminate()
        control.close()
        pool.close()


def _free(port: int, host: str = "0.0.0.0") -> int:
    s = socket.socket()
    try:
        if os.name != "nt":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return port
    except OSError:
        return 0
    finally:
        s.close()


def train(argv):
    p = argparse.ArgumentParser(prog="layerpool train", description="Run a built-in training loop on the pool.")
    p.add_argument("config", nargs="?", help="YAML or JSON file with any of the options below")
    p.add_argument("--model")
    p.add_argument("--mlx-model")
    p.add_argument("--data", help=".jsonl/.json/.txt: chat rows ('messages'), text rows ('text'), or preference rows ('prompt', 'chosen', 'rejected')")
    p.add_argument("--method", choices=["sft", "cpt", "dpo", "ipo", "simpo", "orpo"])
    p.add_argument("--mode", choices=["lora", "full"])
    p.add_argument("--random-init", action="store_true", default=None, help="train from scratch (with --mode full)")
    p.add_argument("--rank", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--schedule", choices=["cosine", "linear", "constant"])
    p.add_argument("--warmup", type=int)
    p.add_argument("--min-lr-ratio", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--epochs", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--seq", type=int)
    p.add_argument("--micro", type=int, help="pieces each batch is cut into, so all devices work at once")
    p.add_argument("--micro-batch-size", type=int, help="rows in each piece, instead of --micro")
    p.add_argument("--target-modules", nargs="+", help="the linear layers LoRA goes on, such as q_proj v_proj (default: all of them)")
    p.add_argument("--save-every", type=int, help="also save a checkpoint folder (out/step-N) every N steps")
    p.add_argument("--resume", help="a checkpoint folder from --save-every: go on from there (same data and settings)")
    p.add_argument("--codec", choices=["int8", "fp16", "fp32"])
    p.add_argument("--ckpt-every", type=int)
    p.add_argument("--max-grad-norm", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--train-on", choices=["assistant", "all"])
    p.add_argument("--out")
    p.add_argument("--name")
    p.add_argument("--local", action="store_true", help="no running pool needed: start one in this process (this machine trains)")
    args = p.parse_args(argv)
    cfg = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            text = f.read()
        try:
            cfg = json.loads(text)
        except ValueError:
            import yaml
            cfg = yaml.safe_load(text)
        cfg = cfg or {}
        if not isinstance(cfg, dict):
            sys.exit(f"layerpool train: {args.config} must hold a mapping of options (model: ..., data: ...)")
        cfg = {str(k).replace("-", "_"): v for k, v in cfg.items()}
        unknown = sorted(set(cfg) - set(TRAIN_KEYS))
        if unknown:
            sys.exit(f"layerpool train: unknown options in {args.config}: {', '.join(unknown)}")
    cfg.update({k: v for k, v in vars(args).items() if k in TRAIN_KEYS and v is not None})
    if not cfg.get("model") or not cfg.get("data"):
        sys.exit("layerpool train: --model and --data are required (or put them in the config file)")
    if isinstance(cfg["data"], str) and not os.path.exists(cfg["data"]):
        sys.exit(f"layerpool train: the data file {cfg['data']} does not exist")
    run_training(cfg, local=args.local)


def run_training(cfg: dict, local: bool = False):
    client = start(control=False) if local else _client()
    method = cfg.get("method", "sft")
    run_kwargs = {k: cfg[k] for k in ("mlx_model", "mode", "rank", "alpha", "lr", "schedule", "warmup", "min_lr_ratio", "weight_decay", "micro", "codec",
                                      "micro_batch_size", "ckpt_every", "random_init", "seq", "target_modules") if cfg.get(k) is not None}
    batch = cfg.get("batch_size", 8)
    if "micro" not in run_kwargs:
        run_kwargs["micro"] = min(8, batch * (2 if method in ("dpo", "ipo", "simpo", "orpo") else 1))
    if local:
        client.wait_for_devices(1, timeout=300)
    run = client.run(cfg["model"], name=cfg.get("name"), **run_kwargs)
    out = cfg.get("out") or f"layerpool-{run.id}"
    print(f"layerpool: run {run.id} started on {len(run.pool.devices())} device(s)", flush=True)
    start_step = 0
    if cfg.get("resume"):
        named = re.search(r"step-(\d+)$", os.path.basename(os.path.normpath(cfg["resume"])))
        if named:
            start_step = int(named.group(1))
        else:
            with open(os.path.join(cfg["resume"], "layerpool_run.json"), encoding="utf-8") as f:
                start_step = json.load(f).get("steps", 0)
        run.load(cfg["resume"])
        print(f"layerpool: going on from step {start_step} ({cfg['resume']})", flush=True)
        saved = os.path.join(cfg["resume"], REFERENCE_FILE)
        reference = np.load(saved) if os.path.exists(saved) else None
    else:
        reference = None
    state = "finished"
    t0, every = time.time(), cfg.get("save_every")

    def on_step(r):
        done = r.get("at", r["step"]) + 1
        total = r.get("total") or 0
        rate = (time.time() - t0) / max(1, done - start_step)
        left = f"  {done}/{total}, about {(total - done) * rate / 60:.0f} min left" if total else ""
        loss = "" if r.get("loss") is None else f"  loss {r['loss']:.4f}"
        print(f"step {done - 1}{loss}  lr {r['lr']:.2e}" + (f"  {r['tokens_per_s']:.0f} tok/s" if r.get("tokens_per_s") else "") + left, flush=True)
        if every and done % every == 0:
            path = run.save(os.path.join(out, f"step-{done}"), optimizer=True)
            if getattr(run, "reference", None) is not None:
                np.save(os.path.join(path, REFERENCE_FILE), run.reference)
            print(f"layerpool: checkpoint {path} (resume with --resume {path})", flush=True)
    try:
        run.train(cfg["data"], method=method, steps=cfg.get("steps"), epochs=cfg.get("epochs", 1.0), batch_size=batch, seq=cfg.get("seq"),
                  max_grad_norm=cfg.get("max_grad_norm", 1.0), beta=cfg.get("beta"), assistant_only=cfg.get("train_on", "assistant") == "assistant",
                  on_step=on_step, start_step=start_step, reference=reference)
        path = run.save(out)
        print(f"layerpool: saved to {path}")
    except BaseException:
        state = "failed"
        raise
    finally:
        run.end(state)
        if local:
            client.close()


def devices(argv):
    p = argparse.ArgumentParser(prog="layerpool devices")
    p.add_argument("--json", action="store_true", help="print every device with its settings and readings as JSON, for scripts")
    args = p.parse_args(argv)
    rows = _client().devices()
    if args.json:
        print(json.dumps(rows, default=str))
        return
    if not rows:
        print("no devices yet: run `layerpool invite` and paste the command on a device")
        return
    print(f"{'id':5} {'name':18} {'status':9} {'layers':9} {'cap':>6} {'memory':>16} {'temp':>5}  hardware")
    for d in rows:
        t = d.get("telemetry", {})
        gpus = t.get("gpus") or []
        mem = f"{sum(g.get('used_gb') or 0 for g in gpus):.1f}/{sum(g.get('total_gb') or 0 for g in gpus):.0f} GB" if gpus else \
            (f"{t.get('ram_used_gb', 0):.1f}/{t.get('ram_total_gb', 0):.0f} GB" if t.get("ram_total_gb") else "n/a")
        temp = t.get("temp_c") or max([g.get("temp_c") or 0 for g in gpus] or [0]) or None
        layers = f"{d['layers'][0]}-{d['layers'][1] - 1}" if d.get("layers") else "-"
        if d.get("layers") and d.get("on_gpu") is not None and d["on_gpu"] < d["layers"][1] - d["layers"][0]:
            layers += "*"
        cap = f"{d['cap_gb']:.1f}" if d.get("cap_gb") else "auto"
        status = {"banned": "left out", "disabled": "paused"}.get(d["status"], d["status"])
        heat = (str(round(temp)) + "C") if temp else ("hot" if str(t.get("heat", "")).startswith("hot") else "-")
        hardware = [f"{i}: {x['name']}" if len(d["devices"]) > 1 else x["name"] for i, x in enumerate(d["devices"])]
        print(f"{d['id']:5} {d['name'][:18]:18} {status:9} {layers:9} {cap:>6} {mem:>16} {heat:>5}  {', '.join(hardware)}")
    if any(d.get("layers") and d.get("on_gpu") is not None and d["on_gpu"] < d["layers"][1] - d["layers"][0] for d in rows):
        print("* some of its layers are in RAM, run by its CPU (--ram-cap)")
    chosen = [d for d in rows if any((d.get("settings") or {}).get(k) is not None for k in ("layers", "gpus", "gpu_caps", "order", "ram_cap"))]
    if chosen:
        print()
        for d in chosen:
            print(f"{d['id']:5} {d['name'][:18]:18} " + describe_settings(d["settings"]))


def set_device(argv):
    p = argparse.ArgumentParser(prog="layerpool set", description="Change a device's settings; memory changes re-split the layers at the next step.")
    p.add_argument("device", help="device id or name")
    p.add_argument("--name")
    p.add_argument("--cap-gb", help="the most memory layerpool may use on this device, in GB: GPU memory on a machine with NVIDIA or AMD GPUs (all the GPUs it trains on together, shared equally unless --gpu-cap sets each), the shared memory on a Mac, RAM on a computer that trains on its CPU. Default: what it can spare (about 92%% of each GPU, 60%% of a Mac's memory, 50%% of RAM); auto clears it")
    p.add_argument("--layers", help="how many layers it takes: a count (8), a share of the model (30%%), a range (4-10) or auto")
    p.add_argument("--gpus", help="the GPUs it trains on, such as 0,2, or auto for all")
    p.add_argument("--gpu-cap", help="the most GPU memory each GPU may use, in GB, such as 0=6,1=10 (GPU numbers as `layerpool devices` "
                                     "lists them); it takes the place of that GPU's share of --cap-gb; auto clears it")
    p.add_argument("--order", help="its place in the chain of devices (lower comes first), or auto")
    p.add_argument("--ram-cap", help="on a machine with GPUs, the RAM in GB it may also use for layers that do not fit its GPUs; its CPU runs those, more slowly, and layerpool gives it only as many as pay off (default: none); auto turns it off")
    args = p.parse_args(argv)
    client = _client()
    wid = _resolve(client, args.device)
    cap = None
    if args.cap_gb is not None:
        try:
            cap = None if args.cap_gb.lower() == "auto" else float(args.cap_gb)
        except ValueError:
            sys.exit("layerpool set: --cap-gb takes a number of GB or 'auto'")
    changes = {"name": args.name} if args.name is not None else {}
    if args.cap_gb is not None:
        changes["cap_gb"] = cap
    if args.ram_cap is not None:
        try:
            changes["ram_cap"] = None if args.ram_cap.lower() == "auto" else float(args.ram_cap)
        except ValueError:
            sys.exit("layerpool set: --ram-cap takes a number of GB or 'auto'")
    for key, value in (("layers", args.layers), ("gpus", args.gpus), ("gpu_caps", args.gpu_cap), ("order", args.order)):
        if value is not None:
            changes[key] = None if value.lower() == "auto" else value
    if not changes:
        sys.exit("layerpool set: nothing to change")
    settings = client.set_device(wid, **changes)
    print(f"layerpool: {settings.get('name') or args.device} now: " + describe_settings(settings))


def describe_settings(settings: dict) -> str:
    caps = settings.get("gpu_caps")
    parts = [f"memory cap {'auto' if settings.get('cap_gb') is None else format(settings['cap_gb'], 'g') + ' GB'}",
             f"layers {settings.get('layers') or 'auto'}"]
    if settings.get("gpus"):
        parts.append("GPUs " + ",".join(str(i) for i in settings["gpus"]))
    if caps:
        parts.append("GPU caps " + ", ".join(f"{k}: {v:g} GB" for k, v in sorted(caps.items())))
    if settings.get("order") is not None:
        parts.append(f"order {settings['order']}")
    if settings.get("ram_cap"):
        parts.append(f"up to {settings['ram_cap']:g} GB of layers in RAM")
    parts.append("in use" if settings.get("enabled", True) else "paused")
    return ", ".join(parts)


def plan_cmd(argv):
    p = argparse.ArgumentParser(prog="layerpool plan", description="Check, before training, whether a model fits the devices in the pool "
                                                                   "now and how its layers would be split.")
    p.add_argument("model", help="Hugging Face model name or local folder")
    p.add_argument("--mlx-model", help="its MLX version, for Macs (a 4-bit one for a 4-bit model)")
    p.add_argument("--full", action="store_true", help="full fine-tuning instead of LoRA")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seq", type=int, default=512, help="tokens per sequence")
    p.add_argument("--micro", type=int, default=8, help="micro-batches per step")
    p.add_argument("--micro-batch-size", type=int, help="sequences per micro-batch")
    p.add_argument("--auto", default="AutoModelForCausalLM", help="the transformers AutoModel class (AutoModelForSeq2SeqLM, ...)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    out = _client().plan(args.model, mlx_model=args.mlx_model, mode="full" if args.full else "lora", rank=args.rank, seq=args.seq,
                         micro=args.micro, micro_batch_size=args.micro_batch_size, auto=args.auto)
    if args.json:
        print(json.dumps(out))
        return
    print(f"{args.model}: {out['layers']} layers of about {out['layer_gb'] or 0:.2f} GB each, with "
          f"{'full fine-tuning' if args.full else f'LoRA rank {args.rank}'}, seq {args.seq}")
    if not out["plan"]:
        print(f"does not fit: {out['why']}")
        sys.exit(1)
    for st in out["plan"]:
        print(f"  {st['name'][:18]:18} layers {st['range'][0]}-{st['range'][1] - 1} ({st['range'][1] - st['range'][0]}), about {st['gb']:.1f} GB")
    print("it fits; the embeddings and output head stay on the computer that runs your training code")


def rebalance(argv):
    p = argparse.ArgumentParser(prog="layerpool rebalance", description="Control how the running run splits its layers.")
    p.add_argument("what", choices=["on", "off", "now"], help="on/off: move layers to faster devices while training, or keep the "
                                                               "split; now: split the layers again at once")
    args = p.parse_args(argv)
    out = _client().set_run(**({"resplit": True} if args.what == "now" else {"rebalance": args.what == "on"}))
    print(f"layerpool: rebalancing is {'on' if out.get('rebalance', True) else 'off'}" + ("; splitting again now" if args.what == "now" else ""))


def _resolve(client, ref: str) -> str:
    devices = [d for d in client.devices() if d.get("status") != "removed"]
    for d in devices:
        if ref == d["id"]:
            return d["id"]
    named = [d["id"] for d in devices if ref == d["name"]]
    if len(named) > 1:
        sys.exit(f"layerpool: several devices are called {ref!r} ({', '.join(named)}); use the id instead")
    if named:
        return named[0]
    sys.exit(f"layerpool: no device called {ref!r}; see `layerpool devices`")


def simple(name: str, argv):
    p = argparse.ArgumentParser(prog=f"layerpool {name}")
    p.add_argument("device")
    args = p.parse_args(argv)
    client = _client()
    wid = _resolve(client, args.device)
    getattr(client, name)(wid)
    print(f"layerpool: {args.device} {dict(pause='paused', resume='back in use', remove='removed')[name]}")


def approve(argv, yes: bool = True):
    name = "approve" if yes else "deny"
    p = argparse.ArgumentParser(prog=f"layerpool {name}", description=f"{name.capitalize()} a device that asked to join (it shows a code).")
    p.add_argument("code", nargs="?", help="the code the device shows; leave it out to list the devices waiting")
    args = p.parse_args(argv)
    pool = _client()._p
    if not args.code:
        rows = pool.waiting()
        if not rows:
            print("no device is waiting. On the device run: layerpool join URL (without --token)")
        for r in rows:
            print(f"{r['code']}  {r['name']:18} {', '.join(r['hardware'])}  from {r.get('from') or '?'}, asked {r['waiting_s']}s ago")
        return
    try:
        who = pool.answer(args.code, approve=yes)
    except Exception as e:
        sys.exit(f"layerpool: {e}")
    print(f"{who} {'approved: it joins in a few seconds' if yes else 'denied'}")


CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "layerpool")


def _size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _tell_if_outdated():
    """A line when PyPI has a newer layerpool (one anonymous request; LAYERPOOL_NO_UPDATE_CHECK=1 turns it off)."""
    if os.environ.get("LAYERPOOL_NO_UPDATE_CHECK"):
        return
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/layerpool/json", timeout=5) as r:
            latest = json.load(r)["info"]["version"]
    except Exception:
        return
    number = lambda v: tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
    if number(latest) > number(__version__):
        print(f"  update      layerpool {latest} is out (you have {__version__}): pip install -U layerpool, here and on every device",
              flush=True)


def end_run(argv):
    argparse.ArgumentParser(prog="layerpool end", description="End the current run, for example one whose script is stuck.").parse_args(argv)
    client = _client()
    run = client.status().get("run")
    if not run:
        print("no run is going")
        return
    client.end_run()
    print(f"ended run {run}; the devices are free")


def cache(argv):
    p = argparse.ArgumentParser(prog="layerpool cache", description="Show or clear what layerpool keeps on this computer's disk.")
    p.add_argument("action", nargs="?", choices=["show", "clear"], default="show")
    args = p.parse_args(argv)
    parts = {"tensors": "weights downloaded from Hugging Face", "ranges": "the parts of models this device held",
             "checkpoints": "checkpoints of pools that did not close cleanly", "store": "checkpoints devices keep while they train"}
    tunnels = [os.path.join(CACHE_ROOT, f) for f in os.listdir(CACHE_ROOT) if f.startswith("cloudflared")] if os.path.isdir(CACHE_ROOT) else []
    if args.action == "show":
        for name, what in parts.items():
            print(f"{_size(os.path.join(CACHE_ROOT, name)) / 1e9:8.2f} GB  {name:12} {what}")
        print(f"{sum(os.path.getsize(t) for t in tunnels) / 1e9:8.2f} GB  {'cloudflared':12} the tunnel program (--tunnel)")
        print(f"\nin {CACHE_ROOT}; `layerpool cache clear` frees it (downloads happen again when needed). Whole model files that "
              "were cheaper to download than their parts are in the Hugging Face cache: `hf cache scan` shows them")
        return
    busy = [e for e in (os.listdir(os.path.join(CACHE_ROOT, "store")) if os.path.isdir(os.path.join(CACHE_ROOT, "store")) else [])
            if e.split("_", 1)[0].isdigit() and psutil.pid_exists(int(e.split("_", 1)[0]))]
    if busy:
        print("a device on this computer is in a pool now, so the downloaded weights and model parts stay; stop it first to free "
              "them too")
    freed = 0
    for path in tunnels:
        try:
            size = os.path.getsize(path)
            os.remove(path)
            freed += size
        except OSError:
            pass
    for name in parts:
        path = os.path.join(CACHE_ROOT, name)
        if not os.path.isdir(path):
            continue
        for entry in os.listdir(path):
            target = os.path.join(path, entry)
            if name == "checkpoints" and time.time() - os.path.getmtime(target) < 3600:
                continue
            if busy and name in ("tensors", "ranges"):
                continue
            if name == "store" and entry.split("_", 1)[0].isdigit() and psutil.pid_exists(int(entry.split("_", 1)[0])):
                continue
            freed += _size(target) if os.path.isdir(target) else os.path.getsize(target)
            shutil.rmtree(target, ignore_errors=True) if os.path.isdir(target) else os.remove(target)
    print(f"freed {freed / 1e9:.2f} GB")


def invite(argv):
    p = argparse.ArgumentParser(prog="layerpool invite", description="Print a join command for a new device.")
    p.add_argument("--reusable", action="store_true", help="one code for several devices (they show up for you to see in the dashboard)")
    p.add_argument("--hours", type=float, default=24)
    args = p.parse_args(argv)
    got = _client().invite(reusable=args.reusable, hours=args.hours)
    print("On a computer:\n  " + got["command"].replace("\n", "\n  "))
    print("\nIn a notebook cell:\n  " + got["notebook"].replace("\n", "\n  "))
    if not got.get("tunnel"):
        print("\nThis pool has no tunnel, so only devices on this network can join. For notebooks (Colab, Kaggle) or devices elsewhere,\n"
              "start the pool with `layerpool up --tunnel`.")


def status(argv):
    p = argparse.ArgumentParser(prog="layerpool status")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("--json", action="store_true", help="print the pool's state as JSON, for scripts")
    args = p.parse_args(argv)
    client = _client()
    if args.json:
        print(json.dumps(client.status(), default=str))
        return
    run, last = object(), None
    while True:
        s = client.status()
        if s["run"] != run:
            run, last = s["run"], None
            print(f"layerpool {s['version']}: {s['devices']} device(s) online, run: {s['run'] or 'none'}")
            for p_ in s.get("plan") or []:
                print(f"  {p_['name']}: layers {p_['range'][0]}-{p_['range'][1] - 1}")
        rows = [h for h in s["history"] if last is None or h["step"] > last]
        for h in rows if last is not None or args.follow else rows[-5:]:
            loss = "  skipped (overflow)" if h.get("skipped") else "" if h.get("loss") is None else f"  loss {h['loss']:.4f}"
            print(f"step {h['step']}{loss}" + (f"  {h['tokens_per_s']:.0f} tok/s" if h.get("tokens_per_s") else ""))
        if s["history"]:
            last = max(last if last is not None else -1, s["history"][-1]["step"])
        if not args.follow:
            return
        time.sleep(2)


COMMANDS = {"cache": cache, "up": up, "join": worker_module.main, "approve": approve, "deny": lambda argv: approve(argv, yes=False), "train": train, "devices": devices, "set": set_device, "invite": invite, "status": status,
            "end": end_run, "rebalance": rebalance, "plan": plan_cmd, "pause": lambda a: simple("pause", a), "resume": lambda a: simple("resume", a), "remove": lambda a: simple("remove", a)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-V", "--version", "version"):
        print(f"layerpool {__version__}")
        return
    if not argv or argv[0] not in COMMANDS:
        if argv and argv[0] not in ("-h", "--help", "help"):
            print(f"layerpool: there is no command {argv[0]!r}\n", file=sys.stderr)
        print(f"layerpool {__version__}: pool the memory of your devices to train one model\n\n"
              "  layerpool up                 start the pool and open the dashboard\n"
              "  layerpool invite             print a join command for a new device\n"
              "  layerpool join URL [--token] lend this device to a pool (without a code it waits for approval)\n"
              "  layerpool approve [CODE]     let a waiting device in (no code: list them); deny CODE refuses it\n"
              "  layerpool devices            list devices, their layers, memory and temperature\n"
              "  layerpool plan MODEL         check whether a model fits the devices, and how it would be split\n"
              "  layerpool set DEVICE ...     change a device's layers, GPUs, memory caps, RAM offload, order or name\n"
              "  layerpool pause / resume     stop using a device, or use it again\n"
              "  layerpool remove DEVICE      take a device out of the pool\n"
              "  layerpool train ...          run a built-in training loop (sft, cpt, dpo, ...)\n"
              "  layerpool status [-f]        follow the current run\n"
              "  layerpool rebalance on|off   move layers to faster devices while training, or keep the split (now: split again)\n"
              "  layerpool end                end the current run (a stuck or abandoned one)\n"
              "  layerpool cache [clear]      show or free the disk space layerpool uses\n\n"
              "In Python: import layerpool; pool = layerpool.connect(); run = pool.run(\"Qwen/Qwen3-0.6B\")")
        sys.exit(0 if not argv or argv[0] in ("-h", "--help", "help") else 2)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    try:
        COMMANDS[argv[0]](argv[1:])
    except KeyboardInterrupt:
        sys.exit(130)
    except (RemoteError, PoolError) as e:
        sys.exit(f"layerpool: {e}")
    except Exception as e:
        sys.exit(f"layerpool: {plain_error(e)}")

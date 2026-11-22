import hashlib
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request

from .codec import replace_file
from .net import ping

VERSION = "2026.10.7"
RELEASES = f"https://github.com/cloudflare/cloudflared/releases/download/{VERSION}/"
SHA256 = {
    "cloudflared-darwin-amd64.tgz": "02f3edfd87684dd5b705607688e4eb4b323f6e777b81e7b3a85af556aa356848",
    "cloudflared-darwin-arm64.tgz": "3586ccf397e98e5f437e60e7c307ca6ac463e110580503af8bb8e6f9ebf831bd",
    "cloudflared-linux-386": "4823d2f038395617e04e4070601ab45c0c43f31f15ee7586be299e19ef49d46e",
    "cloudflared-linux-amd64": "1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872",
    "cloudflared-linux-arm": "74f690f6707182839b63e0465726c7776dd63f0a174390b911d5d71f35cab94e",
    "cloudflared-linux-arm64": "3df43aedc9ff291d1735b5cb576cb97e0b157488796db2e32b28aecabb445775",
    "cloudflared-windows-amd64.exe": "2493dbc168923bb3e59e5e7716f71deced92992eea9acd6e9d74f84ac7751343",
}
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "layerpool")


def _bind_prctl():
    """prctl, looked up once here: a child process after fork must only call it, not import or load anything."""
    if platform.system() != "Linux":
        return None
    try:
        import ctypes
        return ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None


_PRCTL = _bind_prctl()
_TERM = 15


def _die_with_parent():
    _PRCTL(1, _TERM)


def child_options() -> dict:
    """Popen options that end a helper process (cloudflared, systemd-inhibit) when layerpool is killed, on Linux; elsewhere
    layerpool ends them when it exits."""
    return {"preexec_fn": _die_with_parent} if _PRCTL is not None else {}


def _asset() -> str:
    system, machine = platform.system().lower(), platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "arm" if machine.startswith("arm") else "386" if machine in ("i386", "i686", "x86") else "amd64"
    if system == "darwin":
        return f"cloudflared-darwin-{arch}.tgz"
    if system == "windows":
        return "cloudflared-windows-amd64.exe"
    return f"cloudflared-linux-{arch}"


def cloudflared() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    name = f"cloudflared-{VERSION}" + (".exe" if platform.system() == "Windows" else "")
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        return path
    os.makedirs(CACHE, exist_ok=True)
    asset = _asset()
    with tempfile.TemporaryDirectory() as tmp:
        download = os.path.join(tmp, asset)
        with urllib.request.urlopen(RELEASES + asset, timeout=120) as r, open(download, "wb") as f:
            shutil.copyfileobj(r, f)
        with open(download, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != SHA256.get(asset):
            raise RuntimeError(f"the downloaded {asset} does not match cloudflared {VERSION} (sha256 {got}); install cloudflared yourself "
                               "and put it on PATH")
        if asset.endswith(".tgz"):
            with tarfile.open(download) as archive:
                member = next(m for m in archive.getmembers() if os.path.basename(m.name) == "cloudflared" and m.isfile())
                with archive.extractfile(member) as src, open(download + ".bin", "wb") as dst:
                    shutil.copyfileobj(src, dst)
            download += ".bin"
        os.chmod(download, 0o755)
        staged = f"{path}.{os.getpid()}.tmp"
        shutil.copyfile(download, staged)
        os.chmod(staged, 0o755)
        try:
            replace_file(staged, path)
        except OSError:
            os.remove(staged)
            if not os.path.exists(path):
                raise
    return path


class Tunnel:
    def __init__(self, port: int, timeout: float = 60.0, attempts: int = 3, log=None):
        self.port = port
        self.lines: list[str] = []
        self.proc = None
        log = log or (lambda *a: None)
        binary = cloudflared()
        for attempt in range(1, attempts + 1):
            started = time.time()
            log(f"opening a tunnel for port {port} (attempt {attempt} of {attempts}, this can take up to a minute)")
            self.lines = []
            self.proc = subprocess.Popen([binary, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", f"http://127.0.0.1:{port}"],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", **child_options())
            threading.Thread(target=self._read, daemon=True).start()
            try:
                self.url = self._wait_url(timeout)
                self._wait_reachable(timeout)
                log(f"tunnel ready at {self.url} after {time.time() - started:.0f}s")
                return
            except RuntimeError as e:
                log(f"tunnel attempt {attempt} failed: {str(e).splitlines()[0]}")
                if attempt == attempts:
                    raise

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line)
            if len(self.lines) > 400:
                del self.lines[:200]

    def _wait_url(self, timeout: float) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            for line in self.lines:
                found = re.search(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com", line)
                if found:
                    return found.group(0)
            if self.proc.poll() is not None:
                break
            time.sleep(0.2)
        self.close()
        raise RuntimeError("cloudflared did not give a tunnel address:\n" + "".join(self.lines[-20:]))

    def _wait_reachable(self, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if ping(self.url, 10):
                return
            time.sleep(1)
        self.close()
        raise RuntimeError(f"the tunnel {self.url} did not become reachable within {timeout:.0f}s")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

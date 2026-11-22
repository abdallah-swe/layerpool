import hmac
import http.client
import json
import queue
import secrets
import socket
import ssl
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class HTTPError(urllib.error.HTTPError):
    def __init__(self, url: str, code: int, detail: str = ""):
        super().__init__(url, code, detail or f"HTTP {code}", None, None)
        self.detail = detail

    def __str__(self):
        return f"HTTP {self.code}: {self.detail}" if self.detail else f"HTTP {self.code}"


def _detail(body: bytes) -> str:
    text = body.decode("utf-8", "replace").strip()
    try:
        found = json.loads(text)
        if isinstance(found, dict) and found.get("error"):
            return str(found["error"])
    except ValueError:
        pass
    return text[:500]

CHUNK = 16 << 20
MAX_BODY = 256 << 20
UPLOAD_TTL_S = 600


class Cancelled(ConnectionError):
    pass


class Inbox:
    def __init__(self):
        self.items: dict[tuple, bytes] = {}
        self.parts: dict[tuple, dict[int, bytes]] = {}
        self.taken: dict[tuple, None] = {}
        self.cond = threading.Condition()

    def put(self, key: tuple, data: bytes, part: int = 0, parts: int = 1):
        with self.cond:
            if key in self.taken or key in self.items:
                return
            if parts == 1:
                self.items[key] = data
            else:
                got = self.parts.setdefault(key, {})
                got[part] = data
                if len(got) == parts:
                    self.items[key] = b"".join(got[i] for i in range(parts))
                    del self.parts[key]
            self.cond.notify_all()

    def get(self, key: tuple, timeout: float | None = None, cancel=None) -> bytes | None:
        deadline = None if timeout is None else time.time() + timeout
        with self.cond:
            while key not in self.items:
                if cancel is not None and cancel():
                    return None
                left = None if deadline is None else deadline - time.time()
                if left is not None and left <= 0:
                    return None
                self.cond.wait(timeout=0.5 if left is None else min(0.5, left))
            self.taken[key] = None
            if len(self.taken) > 4096:
                del self.taken[next(iter(self.taken))]
            return self.items.pop(key)

    def drop(self, keep):
        with self.cond:
            for key in [k for k in self.items if not keep(k)]:
                del self.items[key]
            for key in [k for k in self.parts if not keep(k)]:
                del self.parts[key]
            for key in [k for k in self.taken if not keep(k)]:
                del self.taken[key]


class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class PeerServer:
    def __init__(self, port: int = 0, token: str | None = None, routes: dict | None = None, host: str = "0.0.0.0", check=None,
                 limits: dict | None = None):
        self.token = token or secrets.token_hex(16)
        self.limits = limits or {}
        self.inbox = Inbox()
        self.routes = routes or {}
        self.uploads: dict[tuple, dict[int, bytes]] = {}
        self.upload_times: dict[tuple, float] = {}
        self.uploads_lock = threading.Lock()
        self.served: dict[tuple, tuple[float, int, bytes]] = {}
        self.check = check
        self.closed = False
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 300

            def log_message(self, *args):
                pass

            def reply(self, code: int, body: bytes = b"", content_type: str | None = None, headers: dict | None = None):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                if content_type:
                    self.send_header("Content-Type", content_type)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def stream(self, code: int, source):
                self.send_response(code)
                self.send_header("Content-Type", getattr(source, "content_type", "text/event-stream"))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    for piece in source:
                        self.wfile.write(piece)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    close = getattr(source, "close", None)
                    if close:
                        close()

            def read_chunked(self, most: int):
                """A body sent in chunks (what tunnels such as Cloudflare's send), whole; "large" past the limit, None when the
                connection broke inside it."""
                parts, total = [], 0
                while True:
                    line = self.rfile.readline(1024)
                    if not line.endswith(b"\n"):
                        return None
                    try:
                        size = int(line.split(b";")[0].strip() or b"0", 16)
                    except ValueError:
                        return None
                    if size == 0:
                        while True:
                            trailer = self.rfile.readline(1024)
                            if trailer in (b"\r\n", b"\n"):
                                return b"".join(parts)
                            if not trailer.endswith(b"\n"):
                                return None
                    total += size
                    if total > most:
                        return "large"
                    data = self.rfile.read(size)
                    if len(data) != size or self.rfile.readline(1024) not in (b"\r\n", b"\n"):
                        return None
                    parts.append(data)

            def handle_any(self, method: str):
                url = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                if server.closed:
                    self.close_connection = True
                    return self.reply(503)
                if url.path == "/ping":
                    return self.reply(200, b"pong")
                if server.check is not None:
                    verdict = server.check(self, method, url.path, query)
                    if verdict is not None:
                        code, out = verdict
                        if isinstance(out, (dict, list)):
                            out = json.dumps(out).encode()
                        self.close_connection = True
                        return self.reply(code, out or b"")
                elif not hmac.compare_digest((self.headers.get("X-Token") or "").encode("utf-8", "surrogateescape"), server.token.encode()):
                    self.close_connection = True
                    return self.reply(403)
                most = server.limits.get(url.path, MAX_BODY)
                if method == "POST" and "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                    body = self.read_chunked(most)
                    if not isinstance(body, bytes):
                        self.close_connection = True
                        return self.reply(413) if body == "large" else None
                else:
                    try:
                        length = int(self.headers.get("Content-Length") or 0)
                    except ValueError:
                        length = -1
                    if length < 0 or length > most:
                        self.close_connection = True
                        return self.reply(413 if length > most else 400)
                    body = self.rfile.read(length) if method == "POST" else b""
                    if len(body) != (length if method == "POST" else 0):
                        self.close_connection = True
                        return
                try:
                    return self.route(method, url, query, body)
                except (KeyError, ValueError) as e:
                    return self.reply(400, json.dumps({"error": repr(e)}).encode())
                except Exception as e:
                    return self.reply(500, json.dumps({"error": repr(e)}).encode())

            def route(self, method: str, url, query: dict, body: bytes):
                if url.path == "/msg" and method == "POST":
                    key = tuple(query["key"].split(","))
                    server.inbox.put(key, body, int(query.get("part", 0)), int(query.get("parts", 1)))
                    return self.reply(200)
                route = server.routes.get((method, url.path))
                if route is None:
                    return self.reply(404)
                with server.uploads_lock:
                    now = time.time()
                    for old in [k for k, t in server.upload_times.items() if now - t > UPLOAD_TTL_S]:
                        server.uploads.pop(old, None)
                        server.upload_times.pop(old, None)
                    for old in [k for k, v in server.served.items() if now - v[0] > 120]:
                        server.served.pop(old, None)
                upload = None
                if method == "POST" and "blob_parts" in query:
                    parts, part = int(query.pop("blob_parts")), int(query.pop("blob_part"))
                    upload = (url.path, tuple(sorted(query.items())))
                    query.pop("blob_id", None)
                    with server.uploads_lock:
                        now = time.time()
                        got = server.uploads.setdefault(upload, {})
                        server.upload_times[upload] = now
                        got[part] = body
                        if len(got) < parts:
                            return self.reply(200)
                        body = b"".join(got[i] for i in range(parts))
                chunk = query.pop("blob_chunk", None)
                sizing = query.pop("blob_size", None)
                cache_key = (url.path, tuple(sorted(query.items())))
                content_type, extra_headers = None, None
                with server.uploads_lock:
                    cached = server.served.get(cache_key) if chunk is not None else None
                    if cached:
                        server.served[cache_key] = (time.time(), cached[1], cached[2])
                if cached:
                    code, out = cached[1], cached[2]
                else:
                    answer = route(query, body)
                    code, out = answer[0], answer[1]
                    content_type = answer[2] if len(answer) > 2 else None
                    extra_headers = answer[3] if len(answer) > 3 else None
                if upload is not None:
                    with server.uploads_lock:
                        if code == 200:
                            server.uploads.pop(upload, None)
                            server.upload_times.pop(upload, None)
                if hasattr(out, "content_type") and hasattr(out, "__iter__"):
                    return self.stream(code, out)
                if isinstance(out, (dict, list)):
                    out = json.dumps(out).encode()
                out = out or b""
                if code == 200 and sizing is not None:
                    if len(out) > CHUNK:
                        with server.uploads_lock:
                            server.served[cache_key] = (time.time(), code, out)
                    return self.reply(200, json.dumps({"size": len(out)}).encode())
                if code == 200 and chunk is not None:
                    if cached and (int(chunk) + 1) * CHUNK >= len(out):
                        with server.uploads_lock:
                            server.served.pop(cache_key, None)
                    out = out[int(chunk) * CHUNK:(int(chunk) + 1) * CHUNK]
                return self.reply(code, out, content_type, extra_headers)

            def do_GET(self):
                self.handle_any("GET")

            def do_POST(self):
                self.handle_any("POST")

        self.httpd = _Server((host, port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def lan_url(self) -> str:
        return f"http://{lan_ip()}:{self.port}"

    def url_at(self, address: str) -> str:
        """An address given by hand (http://10.0.0.5:7400, or just 10.0.0.5 for this server's own port) as a full URL."""
        address = address if "://" in address else f"http://{address}"
        parsed = urlparse(address)
        return address if parsed.port or parsed.scheme == "https" else f"{address.rstrip('/')}:{self.port}"

    def close(self):
        self.closed = True
        self.httpd.shutdown()
        self.httpd.server_close()


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


_resolved: dict[str, tuple[str, float]] = {}


def _ask_public_dns(host: str) -> str | None:
    for server, path in (("1.1.1.1", "/dns-query"), ("8.8.8.8", "/resolve")):
        try:
            conn = http.client.HTTPSConnection(server, timeout=5, context=ssl.create_default_context())
            conn.request("GET", f"{path}?name={host}&type=A", headers={"Accept": "application/dns-json"})
            answer = json.loads(conn.getresponse().read())
            conn.close()
            for record in answer.get("Answer") or []:
                if record.get("type") == 1:
                    return record["data"]
        except Exception:
            continue
    return None


def resolve(host: str) -> str:
    """The address to connect to. A fresh tunnel name is often cached as missing by the local resolver for many
    minutes, so when the system cannot find a name, ask public DNS over HTTPS instead."""
    try:
        socket.getaddrinfo(host, None)
        return host
    except socket.gaierror:
        cached = _resolved.get(host)
        if cached and cached[1] > time.time():
            return cached[0]
        ip = _ask_public_dns(host)
        if ip is None:
            raise
        _resolved[host] = (ip, time.time() + 300)
        return ip


class _HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, port=None, timeout=None, context=None):
        super().__init__(host, port, timeout=timeout, context=context)
        self.tls = context

    def connect(self):
        sock = socket.create_connection((resolve(self.host), self.port or 443), self.timeout, self.source_address)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = self.tls.wrap_socket(sock, server_hostname=self.host)


def connection(url: str, timeout: float):
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return _HTTPSConnection(parsed.hostname, parsed.port, timeout=timeout, context=ssl.create_default_context())
    return http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)


def ping(url: str, timeout: float = 10) -> bool:
    conn = connection(url, timeout)
    try:
        conn.request("GET", "/ping")
        r = conn.getresponse()
        return r.status == 200 and r.read() == b"pong"
    except Exception:
        return False
    finally:
        conn.close()


class Peer:
    def __init__(self, urls: list[str], token: str):
        self.urls = [u for u in urls if u]
        self.token = token
        self.url = None
        self.local = threading.local()

    def _connection(self, url: str, timeout: float):
        conns = getattr(self.local, "conns", None)
        if conns is None:
            conns = self.local.conns = {}
        conn = conns.get(url)
        if conn is None:
            conn = conns[url] = connection(url, timeout)
        conn.timeout = timeout
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
        return conn

    def _drop(self, url: str):
        conn = getattr(self.local, "conns", {}).pop(url, None)
        if conn is not None:
            conn.close()

    def pick(self) -> str:
        if self.url:
            return self.url
        for u in self.urls:
            if ping(u, 2 if u.startswith("http://") else 15):
                self.url = u
                return u
        raise ConnectionError(f"no address of this peer answers: {self.urls}")

    def request(self, path: str, data: bytes | None = None, timeout: float = 90, retries: int = 5, cancel=None) -> bytes:
        last = None
        for attempt in range(retries):
            if cancel is not None and cancel():
                raise Cancelled(path)
            url = None
            try:
                url = self.pick()
                conn = self._connection(url, timeout)
                conn.request("POST" if data is not None else "GET", path, body=data, headers={"X-Token": self.token or "", "Content-Length": str(len(data or b""))})
                r = conn.getresponse()
                body = r.read()
                if r.status >= 400:
                    raise HTTPError(url + path, r.status, _detail(body))
                return body
            except urllib.error.HTTPError as e:
                if e.code in (400, 403, 404, 409, 410, 413):
                    raise
                last = e
                if url:
                    self._drop(url)
            except Exception as e:
                last = e
                if url:
                    self._drop(url)
                self.url = None
            if attempt == retries - 1:
                break
            pause = time.time() + min(2 ** attempt, 10)
            while time.time() < pause and not (cancel is not None and cancel()):
                time.sleep(0.1)
        raise ConnectionError(f"request {path} failed: {last!r}")

    def send(self, key: tuple, data: bytes, cancel=None, retries: int = 5):
        k = ",".join(str(x) for x in key)
        if len(data) <= CHUNK:
            self.request(f"/msg?key={k}", data, cancel=cancel, retries=retries)
            return
        parts = (len(data) + CHUNK - 1) // CHUNK
        view = memoryview(data)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(self.request, f"/msg?key={k}&part={i}&parts={parts}", view[i * CHUNK:(i + 1) * CHUNK], cancel=cancel,
                                       retries=retries) for i in range(parts)]:
                future.result()

    def post_blob(self, path: str, data: bytes, timeout: float = 120, retries: int = 5, cancel=None):
        if len(data) <= CHUNK:
            return self.request(path, data, timeout=timeout, retries=retries, cancel=cancel)
        parts = (len(data) + CHUNK - 1) // CHUNK
        sep = "&" if "?" in path else "?"
        blob = secrets.token_hex(6)
        view = memoryview(data)
        for i in range(parts):
            self.request(f"{path}{sep}blob_id={blob}&blob_part={i}&blob_parts={parts}", view[i * CHUNK:(i + 1) * CHUNK], timeout=timeout,
                         retries=retries, cancel=cancel)
        return b""

    def get_blob(self, path: str, timeout: float = 120, retries: int = 5) -> bytes:
        sep = "&" if "?" in path else "?"
        size = json.loads(self.request(f"{path}{sep}blob_size=1", timeout=timeout, retries=retries))["size"]
        if size <= CHUNK:
            return self.request(path, timeout=timeout, retries=retries)
        return b"".join(self.request(f"{path}{sep}blob_chunk={i}", timeout=timeout, retries=retries) for i in range((size + CHUNK - 1) // CHUNK))

    def json(self, path: str, payload=None, timeout: float = 90):
        body = None if payload is None else json.dumps(payload).encode()
        out = self.request(path, body, timeout=timeout)
        return json.loads(out) if out else None


class Sender:
    def __init__(self, peer: Peer, streams: int = 3, cancel=None):
        self.peer, self.cancel = peer, cancel
        self.jobs: queue.Queue = queue.Queue()
        self.errors: list[Exception] = []
        self.busy = 0.0
        self.lock = threading.Lock()
        self.threads = [threading.Thread(target=self._run, daemon=True) for _ in range(streams)]
        [t.start() for t in self.threads]

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                self.jobs.task_done()
                return
            key, data = job
            t = time.time()
            try:
                if not self.errors:
                    self.peer.send(key, data, cancel=self.cancel)
            except Exception as e:
                self.errors.append(e)
            finally:
                with self.lock:
                    self.busy += time.time() - t
                self.jobs.task_done()

    def put(self, key: tuple, data: bytes):
        if self.errors:
            raise self.errors[0]
        self.jobs.put((key, data))

    def flush(self):
        self.jobs.join()
        if self.errors:
            raise self.errors[0]

    def close(self):
        for _ in self.threads:
            self.jobs.put(None)

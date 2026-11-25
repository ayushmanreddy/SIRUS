from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BACKEND = "http://127.0.0.1:8000"
MAX_BODY_BYTES = 12 * 1024 * 1024
DOCUMENT_ID = re.compile(r"^/documents/[^/]+$")


def allowed(method: str, path: str) -> bool:
    if method == "GET":
        return path in {
            "/health/live",
            "/health/ready",
            "/system/status",
            "/system/version",
            "/documents",
        } or bool(DOCUMENT_ID.fullmatch(path))
    if method == "POST":
        return path in {"/documents/ingest", "/documents/query"}
    return False


class Handler(BaseHTTPRequestHandler):
    server_version = "SIRUSLocalSite/1.0"

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        page = {
            "/": "index.html",
            "/index.html": "index.html",
            "/sirus-ui.html": "sirus-ui.html",
        }.get(parsed.path)
        if page:
            try:
                body = (ROOT / page).read_bytes()
            except OSError:
                self.send_error(500, "The SIRUS interface file is missing")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if parsed.path.startswith("/api/"):
            self._proxy("GET", parsed)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path.startswith("/api/"):
            self._proxy("POST", parsed)
            return
        self.send_error(404, "Not found")

    def _proxy(self, method: str, parsed) -> None:
        path = parsed.path[len("/api") :]
        if not allowed(method, path):
            self.send_error(404, "API route is not available through this local interface")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid content length")
            return
        if length > MAX_BODY_BYTES:
            self.send_error(413, "Request exceeds the local interface size limit")
            return
        body = self.rfile.read(length) if length else None
        url = BACKEND + path + (("?" + parsed.query) if parsed.query else "")
        headers = {}
        content_type = self.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=180) as response:
                response_body = response.read()
                status = response.status
                response_type = response.headers.get("Content-Type", "application/json")
        except HTTPError as error:
            response_body = error.read()
            status = error.code
            response_type = error.headers.get("Content-Type", "application/json")
        except (URLError, TimeoutError, OSError) as error:
            response_body = json.dumps(
                {"detail": "Could not reach SIRUS at 127.0.0.1:8000.", "error": str(error)}
            ).encode("utf-8")
            status = 502
            response_type = "application/json; charset=utf-8"
        self.send_response(status)
        self.send_header("Content-Type", response_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8080), Handler)
    print("SIRUS local interface: http://127.0.0.1:8080")
    print("Bound to this computer only. Keep this window open.")
    print("SIRUS API should be running at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the SIRUS local interface.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

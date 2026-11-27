#!/usr/bin/env python3
"""Serve the built site locally with HTTP Range support (single ranges), as
static hosts do, so that video seeking can be checked before deployment.

  tools/serve_site.py [DIR] [--port 8792]

Binds to 127.0.0.1 only.
"""

import argparse
import functools
import http.server
import os
import re


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    _remaining = None
    _accept_sent = False

    def send_head(self):
        self._accept_sent = False
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
        size = os.path.getsize(path)
        if not m or (m.group(1) == "" and m.group(2) == ""):
            self.send_error(416)
            return None
        if m.group(1) == "":
            start, end = max(0, size - int(m.group(2))), size - 1
        else:
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self._accept_sent = True
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(source, outputfile)
        while n > 0:
            chunk = source.read(min(65536, n))
            if not chunk:
                break
            outputfile.write(chunk)
            n -= len(chunk)
        self._remaining = None

    def end_headers(self):
        if not self._accept_sent:
            self.send_header("Accept-Ranges", "bytes")
            self._accept_sent = True
        super().end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="dist/site")
    ap.add_argument("--port", type=int, default=8792)
    a = ap.parse_args()
    handler = functools.partial(RangeHandler, directory=a.dir)
    with http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler) as srv:
        print(f"serving {a.dir} on http://127.0.0.1:{a.port}/", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()

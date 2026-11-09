#!/usr/bin/env python3
"""Read native Meta HTTP JSON without depending on connection-close framing."""

import argparse
import json
import socket
import time
from pathlib import Path


def _unchunk(data):
    chunks = []
    offset = 0
    while True:
        line_end = data.find(b"\r\n", offset)
        if line_end < 0:
            return None
        length = int(data[offset:line_end].split(b";")[0], 16)
        offset = line_end + 2
        if length == 0:
            return b"".join(chunks)
        if len(data) < offset + length + 2:
            return None
        chunks.append(data[offset : offset + length])
        offset += length + 2


def get_json(host="localhost", port=17000, path="/api/v1/capacity/usage", timeout=8):
    started = time.time()
    data = b""
    with socket.create_connection((host, port), timeout) as client:
        client.settimeout(timeout)
        client.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: keep-alive\r\nAccept: application/json\r\n\r\n".encode()
        )
        while len(data) < 16 * 1024 * 1024:
            chunk = client.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\r\n\r\n" not in data:
                continue
            headers, body = data.split(b"\r\n\r\n", 1)
            if b"transfer-encoding: chunked" in headers.lower():
                body = _unchunk(body)
                if body is None:
                    continue
            try:
                value, end = json.JSONDecoder().raw_decode(body.decode().lstrip())
                return dict(
                    status=int(headers.split(b" ")[1]),
                    body=value,
                    seconds=time.time() - started,
                    path=path,
                )
            except (ValueError, UnicodeDecodeError):
                continue
    raise RuntimeError(f"No complete JSON response for {path}: {data[:512]!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=17000)
    p.add_argument("--output", type=Path)
    p.add_argument("--path", default="/api/v1/capacity/usage")
    a = p.parse_args()
    result = get_json(host=a.host, port=a.port, path=a.path)
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)

#!/usr/bin/env python3
"""Bring a remote board's ENTRY events to a local snap.py.

The Arduino is plugged into someone else's laptop. Rather than installing the
whole camera stack over there, we read their serial port over SSH and fire our
own /snap. Nothing is installed on the remote box — it only needs `cat`.

    python3 bridge.py kartiksangani@10.32.1.165 /dev/cu.usbmodem30088037032

Deliberately separate from snap.py: snap.py holds warm models and an open
camera, and restarting it costs seconds. This can be killed and restarted
freely without disturbing it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request

TRIGGERS = ("ENTRY", ">P I")


def fire(url: str) -> None:
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            print(f"    -> {r.read().decode().strip()}")
    except Exception as e:
        print(f"    ! snap failed: {e}")


def run_once(host: str, port: str, baud: int, url: str) -> None:
    """One SSH session, streaming until it drops."""
    remote = f"stty -f {port} {baud} raw -echo; exec cat {port}"
    proc = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         "-o", "ServerAliveInterval=5", host, remote],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, text=True,
    )
    print(f"connected — {host}:{port} @ {baud}")
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith(TRIGGERS):
            print(f"[{time.strftime('%H:%M:%S')}] {line}")
            fire(url)
        else:
            print(f"  mcu: {line}")
    err = (proc.stderr.read() if proc.stderr else "").strip()
    raise RuntimeError(err or "ssh stream ended")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("host", help="user@host of the machine the board is plugged into")
    ap.add_argument("port", help="remote serial device, e.g. /dev/cu.usbmodem301")
    ap.add_argument("--baud", type=int, default=9600,
                    help="sketch is Serial.begin(9600) today (default %(default)s)")
    ap.add_argument("--url", default="http://localhost:8200/snap")
    args = ap.parse_args()

    print(f"bridging {args.host}:{args.port} -> {args.url}\n")
    while True:
        try:
            run_once(args.host, args.port, args.baud, args.url)
        except KeyboardInterrupt:
            print("\nbye")
            return 0
        except Exception as e:
            print(f"! {e} — reconnecting in 3s")
            time.sleep(3)


if __name__ == "__main__":
    sys.exit(main())

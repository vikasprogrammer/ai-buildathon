#!/usr/bin/env python3
"""Take a photo when the door says someone came in.

Deliberately standalone: no zones, no calibration, no tracking, no pipeline.
It holds the camera open and writes a JPEG when something tells it to.

    python3 snap.py                              # starts, listens on :8200
    curl -X POST localhost:8200/snap             # take a photo now
    python3 snap.py --serial /dev/tty.usbmodem*  # ...and fire on the MCU's ">P I"

The camera does not connect to the Arduino. It connects here. The MCU only sends
a line of text saying someone came in; this reads that line and takes the photo.
Photos land in data/snaps/ named by timestamp.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

OUT = "data/snaps"
CAM = 0
_lock = threading.Lock()
_cap = None
_count = 0


def open_camera(index):
    """Hold the camera open. Opening per-shot costs ~1s and drops the moment."""
    global _cap
    _cap = cv2.VideoCapture(index)
    if not _cap.isOpened():
        raise SystemExit(f"could not open camera {index}")
    # Warm up: the first few frames off a webcam are usually dark or green.
    for _ in range(5):
        _cap.read()
    print(f"camera {index} open")


def snap(reason="manual"):
    """Grab the current frame and write it. Returns the path, or None."""
    global _count
    with _lock:
        ok, frame = _cap.read()
        if not ok or frame is None:
            print("! camera gave no frame")
            return None
        os.makedirs(OUT, exist_ok=True)
        name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        path = os.path.join(OUT, f"{name}_{reason}.jpg")
        cv2.imwrite(path, frame)
        _count += 1
        print(f"[{_count}] {path}")
        return path


def watch_serial(port, baud=115200):
    """Snap when the MCU reports an inward passage (`>P I 84`).

    Reconnects on its own — at a venue the board gets unplugged, and the whole
    demo dying because of a loose USB cable is not a good look.
    """
    import serial  # only needed on this path

    while True:
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                print(f"serial {port} open @ {baud}")
                while True:
                    line = ser.readline().decode("ascii", "replace").strip()
                    if not line:
                        continue
                    # The visitor_counter sketch prints "ENTRY  |  Occupancy: 3 ...".
                    # ">P I" is the older spec'd form — accept both so we work
                    # whichever way the firmware ends up going.
                    if line.startswith("ENTRY") or line.startswith(">P I"):
                        snap("in")
                    else:
                        print(f"  mcu: {line}")  # eyeball the other events
        except Exception as e:
            print(f"! serial: {e} — retrying in 2s")
            time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") != "/snap":
            self.send_error(404)
            return
        path = snap("in")
        body = f'{{"ok": {"true" if path else "false"}, "path": "{path or ""}"}}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(f'{{"snaps": {_count}, "dir": "{OUT}"}}'.encode())

    def log_message(self, *a):
        pass  # the snap line is the only output worth having


def main():
    global OUT, CAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--out", default="data/snaps")
    ap.add_argument("--serial", help="MCU serial port, e.g. /dev/tty.usbmodem1101")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    OUT, CAM = args.out, args.camera
    open_camera(CAM)

    if args.serial:
        threading.Thread(
            target=watch_serial, args=(args.serial, args.baud), daemon=True
        ).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"ready — POST http://localhost:{args.port}/snap")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{_count} photos in {OUT}")
    finally:
        if _cap:
            _cap.release()


if __name__ == "__main__":
    main()

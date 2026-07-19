"""
WiFi Listener Brick — Arduino UNO Q + App Lab
---------------------------------------------
Polls the ESP32-C6 (Glyph) HTTP status endpoint over WiFi and forwards
each change to the MCU sketch by calling its RPCs through Arduino Bridge.

Expected response body (JSON) from GET http://<ESP_IP>/status:
    {"access": true, "uid": "6125F917", "count": 2, "button_seq": 3}

Fields:
    access      True after an authorized card has been tapped
    uid         Hex string of the last card scanned (empty until first tap)
    count       Encoder position 0..9 (absolute)
    button_seq  Monotonically-increasing button press counter

Each poll we diff against the previous poll:
    count changes           -> Bridge.call("on_count", new_value)
    button_seq increases    -> Bridge.call("on_button") once per increment
    uid changes             -> Bridge.call("on_rfid", uid)

The MCU sketch:
    * Treats each on_button() as a MODE TOGGLE (normal <-> admin).
    * In admin mode, uses on_count() to update the threshold on the display.
    * When exiting admin, the last count value becomes the new threshold.
"""

import json
import socket
import threading
import time
import urllib.error
import urllib.request

from arduino.app_utils import App, Bridge


# ====== CONFIG ======================================================
ESP_URL          = "http://10.32.1.132/status"
POLL_INTERVAL_S  = 0.15   # 150 ms
HTTP_TIMEOUT_S   = 1.0
VERBOSE_EVERY_N  = 20     # print raw state once every N successful polls even if unchanged
# ====================================================================


prev_count      = None
prev_button_seq = None
prev_uid        = None


def print_network_info() -> None:
    print("---- network info ----", flush=True)
    try:
        print(f"hostname: {socket.gethostname()}", flush=True)
    except Exception as e:
        print(f"hostname failed: {e}", flush=True)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        print(f"local IP: {s.getsockname()[0]}", flush=True)
        s.close()
    except Exception as e:
        print(f"local IP lookup failed: {e}", flush=True)
    print("----------------------", flush=True)


def fetch_status() -> dict | None:
    try:
        with urllib.request.urlopen(ESP_URL, timeout=HTTP_TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)
    except urllib.error.URLError as e:
        print(f"[HTTP] fetch failed: {e}", flush=True)
        return None
    except json.JSONDecodeError as e:
        print(f"[HTTP] bad JSON: {e}", flush=True)
        return None
    except Exception as e:
        print(f"[HTTP] unexpected error: {e}", flush=True)
        return None


def apply_status(data: dict, tick: int) -> None:
    global prev_count, prev_button_seq, prev_uid

    try:
        count      = int(data["count"])
        button_seq = int(data["button_seq"])
        uid        = str(data.get("uid", ""))
        access     = bool(data.get("access", False))
    except (KeyError, ValueError) as e:
        print(f"[HTTP] bad payload shape: {data!r} ({e})", flush=True)
        return

    # Periodically print raw state so silence isn't ambiguous during debugging
    if tick % VERBOSE_EVERY_N == 0:
        print(f"[poll #{tick}] access={access} uid={uid!r} count={count} button_seq={button_seq}",
              flush=True)

    # First reading -> snapshot only, and push initial count to the MCU
    if prev_count is None:
        prev_count      = count
        prev_button_seq = button_seq
        prev_uid        = uid
        Bridge.call("on_count", count)
        print(f"[INIT] state snapshot: count={count}, button_seq={button_seq}, uid={uid!r}",
              flush=True)
        return

    # Count changed -> push new value straight to MCU (it uses it as threshold in admin mode)
    if count != prev_count:
        Bridge.call("on_count", count)
        print(f"[EVT] count {prev_count} -> {count}   ==> on_count({count})", flush=True)
        prev_count = count

    # Button pressed one or more times -> one on_button per press (each toggles mode)
    if button_seq != prev_button_seq:
        n = button_seq - prev_button_seq
        if n < 0:
            print(f"[EVT] button_seq reset {prev_button_seq} -> {button_seq}, resyncing", flush=True)
        else:
            for _ in range(n):
                Bridge.call("on_button")
            print(f"[EVT] button pressed x{n}   ==> on_button()", flush=True)
        prev_button_seq = button_seq

    # New card scanned -> forward UID (per-tap detection is limited to UID changes)
    if uid and uid != prev_uid:
        Bridge.call("on_rfid", uid)
        print(f"[EVT] rfid {uid!r} access={access}   ==> on_rfid()", flush=True)
        prev_uid = uid


def poll_loop() -> None:
    print(f"[BOOT] HTTP poller ready. GET {ESP_URL} every {POLL_INTERVAL_S:.2f} s.", flush=True)
    tick = 0
    while True:
        tick += 1
        data = fetch_status()
        if data is not None:
            apply_status(data, tick)
        time.sleep(POLL_INTERVAL_S)


def heartbeat() -> None:
    # Slower heartbeat so the log doesn't fill up with alive lines
    while True:
        time.sleep(60)
        print("[heartbeat] brick alive", flush=True)


print_network_info()

threading.Thread(target=poll_loop, daemon=True).start()
threading.Thread(target=heartbeat, daemon=True).start()

print("[BOOT] WiFi listener brick ready.", flush=True)

App.run()

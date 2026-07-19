"""
WiFi Listener Brick — Arduino UNO Q + App Lab
---------------------------------------------
Polls the ESP32-C6 (Glyph) HTTP status endpoint over WiFi and forwards
any change to the MCU sketch by calling its RPCs through Arduino Bridge.

Expected response body (JSON) from GET http://<ESP_IP>/status:
    {"access": true, "uid": "6125F917", "count": 2, "button_seq": 3}

Fields:
    access      True after an authorized card has been tapped
    uid         Hex string of the last card scanned (empty until first tap)
    count       Encoder position 0..N (absolute)
    button_seq  Monotonically-increasing button press counter

Each poll we diff against the previous poll's values:
    count changes           -> Bridge.call("on_encoder", +/-1) per unit
    button_seq increases    -> Bridge.call("on_button") once per increment
    uid changes             -> Bridge.call("on_rfid", uid)

The MCU sketch registers on_encoder / on_button / on_rfid with Bridge.provide()
and prints "[link] ..." lines to Serial for every RPC that arrives, so all
activity is visible on App Lab's Serial Monitor tab.
"""

import json
import socket
import threading
import time
import urllib.error
import urllib.request

from arduino.app_utils import App, Bridge


# ====== CONFIG ======================================================
# Change ESP_URL to whatever IP the Glyph prints on its own Serial Monitor.
ESP_URL          = "http://10.32.1.132/status"
POLL_INTERVAL_S  = 0.15   # 150 ms -- responsive without hammering the ESP32
HTTP_TIMEOUT_S   = 1.0
# ====================================================================


# Previous poll's values (None until we've seen the first successful fetch)
prev_count      = None
prev_button_seq = None
prev_uid        = None


def print_network_info() -> None:
    """Log this container's local IP so we can compare it to the ESP's IP."""
    print("---- network info ----", flush=True)
    try:
        print(f"hostname: {socket.gethostname()}", flush=True)
    except Exception as e:
        print(f"hostname failed: {e}", flush=True)
    try:
        # Connect a UDP socket to an external IP (no packet is sent) so the
        # OS fills in the local IP it would have used.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        print(f"local IP: {s.getsockname()[0]}", flush=True)
        s.close()
    except Exception as e:
        print(f"local IP lookup failed: {e}", flush=True)
    print("----------------------", flush=True)


def fetch_status() -> dict | None:
    """GET the /status endpoint and return the parsed JSON, or None on failure."""
    try:
        with urllib.request.urlopen(ESP_URL, timeout=HTTP_TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)
    except urllib.error.URLError as e:
        print(f"HTTP fetch failed: {e}", flush=True)
        return None
    except json.JSONDecodeError as e:
        print(f"bad JSON from ESP: {e}", flush=True)
        return None


def apply_status(data: dict) -> None:
    """Diff against the previous poll and fire an RPC for anything that changed."""
    global prev_count, prev_button_seq, prev_uid

    try:
        count      = int(data["count"])
        button_seq = int(data["button_seq"])
        uid        = str(data.get("uid", ""))
        access     = bool(data.get("access", False))
    except (KeyError, ValueError) as e:
        print(f"bad payload shape: {data!r} ({e})", flush=True)
        return

    # First reading -> snapshot only, don't fire spurious events
    if prev_count is None:
        prev_count      = count
        prev_button_seq = button_seq
        prev_uid        = uid
        print(f"initial state: count={count}, button_seq={button_seq}, uid={uid!r}, access={access}",
              flush=True)
        return

    # ---- Encoder ----
    if count != prev_count:
        delta = count - prev_count
        step  = 1 if delta > 0 else -1
        for _ in range(abs(delta)):
            Bridge.call("on_encoder", step)
        print(f"encoder {delta:+d}  -> on_encoder({step}) x{abs(delta)}", flush=True)
        prev_count = count

    # ---- Button ----
    if button_seq != prev_button_seq:
        n = button_seq - prev_button_seq
        if n < 0:
            # ESP32 restarted; the seq counter reset. Resync silently.
            print(f"button_seq reset {prev_button_seq} -> {button_seq}, resyncing", flush=True)
        else:
            for _ in range(n):
                Bridge.call("on_button")
            print(f"button pressed x{n}", flush=True)
        prev_button_seq = button_seq

    # ---- RFID ----
    # Fires on any change of the UID field. Same card tapped twice in a row
    # won't refire (limitation of polling a "last-value" field).
    if uid and uid != prev_uid:
        Bridge.call("on_rfid", uid)
        print(f"RFID {uid!r}  access={access}  -> on_rfid()", flush=True)
        prev_uid = uid


def poll_loop() -> None:
    print(f"HTTP poller ready. GET {ESP_URL} every {POLL_INTERVAL_S:.2f} s.", flush=True)
    while True:
        data = fetch_status()
        if data is not None:
            apply_status(data)
        time.sleep(POLL_INTERVAL_S)


def heartbeat() -> None:
    while True:
        time.sleep(10)
        print("[heartbeat] brick alive", flush=True)


print_network_info()

threading.Thread(target=poll_loop,  daemon=True).start()
threading.Thread(target=heartbeat,  daemon=True).start()

print("WiFi listener brick ready.", flush=True)

# App.run() keeps the process alive and services the Bridge RPC bus
App.run()

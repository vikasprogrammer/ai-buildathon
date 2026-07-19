"""
WiFi Listener Brick — Arduino UNO Q + App Lab
---------------------------------------------
Runs on the Uno Q's Linux side. Listens on the local WiFi for UDP packets
broadcast by the ESP32-C6 (Glyph), and forwards each event to the MCU
sketch by calling its RPCs through Arduino Bridge.

Message protocol (from the Glyph):
    +          -> Bridge.call("on_encoder", +1)
    -          -> Bridge.call("on_encoder", -1)
    P          -> Bridge.call("on_button")
    U:<hex>    -> Bridge.call("on_rfid", "<hex>")

The MCU sketch registers on_encoder / on_button / on_rfid with Bridge.provide()
in its setup(). No UART, no permissions to fight with.

Setup:
    1. Make sure the Glyph and the Uno Q are on the same WiFi network.
    2. UDP_PORT must match the ESP32 sketch (default 4210).
    3. Nothing else -- Arduino.app_utils and Bridge are provided by App Lab.
"""

import socket
import sys
import threading

from arduino.app_utils import App, Bridge

UDP_PORT = 4210  # must match WIFI_UDP_PORT in the ESP32 sketch


def dispatch(line: str, sender: str) -> None:
    """Turn one text line from the Glyph into one Bridge RPC call."""
    if line == "+":
        Bridge.call("on_encoder", 1)
        print(f"from {sender}: + -> on_encoder(+1)")
    elif line == "-":
        Bridge.call("on_encoder", -1)
        print(f"from {sender}: - -> on_encoder(-1)")
    elif line == "P":
        Bridge.call("on_button")
        print(f"from {sender}: P -> on_button()")
    elif line.startswith("U:"):
        uid = line[2:]
        Bridge.call("on_rfid", uid)
        print(f"from {sender}: U:{uid} -> on_rfid({uid!r})")
    else:
        print(f"from {sender}: {line!r}  (unknown, ignored)")


def udp_listener() -> None:
    """Background thread: bind a UDP socket and forward every packet to the MCU."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"UDP listening on port {UDP_PORT} (broadcast). Waiting for the Glyph...")

    while True:
        try:
            data, addr = sock.recvfrom(256)
            if not data:
                continue
            text = data.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            dispatch(text, addr[0])
        except Exception as e:
            print(f"UDP loop error: {e}", file=sys.stderr)


threading.Thread(target=udp_listener, daemon=True).start()

print("WiFi listener brick ready.")

# App.run() keeps the process alive and services the Bridge RPC bus
App.run()

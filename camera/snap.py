#!/usr/bin/env python3
"""Take a photo when the door says someone came in — and say who it was.

It holds the camera open and writes a JPEG when something tells it to. If a
face gallery is configured, it also identifies whoever crossed and alerts.

    python3 snap.py                              # starts, listens on :8200
    curl -X POST localhost:8200/snap             # take a photo now
    python3 snap.py --serial /dev/tty.usbmodem*  # ...and fire on the MCU's ">P I"
    open http://localhost:8200/                  # enrol faces, watch the live feed

The camera does not connect to the Arduino. It connects here. The MCU only sends
a line of text saying someone came in; this reads that line and takes the photo.
Photos land in data/snaps/ named by timestamp.

Recognition is deliberately *off the trigger path*. A crossing captures a short
burst and returns immediately; identification happens on a worker thread and is
published as an event. The doorway never waits on a model. That is also what
makes this feasible on the UNO Q — we run a few inferences per crossing, not
thirty per second.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

# ---------------------------------------------------------------------------
# Config — tunable at the venue without hunting through the code
# ---------------------------------------------------------------------------

OUT = "data/snaps"
UNKNOWN_DIR = "data/unknown"
CAM = 0

#: Frames grabbed per crossing. One frame at the moment of crossing is a coin
#: flip: motion blur, a turned head, a blink. Three costs ~66 ms of camera time
#: and turns most coin flips into hits.
BURST = 3

#: Cap on queued recognition jobs. If crossings outrun the model we drop and say
#: so — a silent backlog would report the wrong person for the wrong crossing,
#: which is worse than admitting the miss.
MAX_PENDING = 4

#: Events retained for the UI's initial paint.
HISTORY = 60

_lock = threading.Lock()
_cap = None
_count = 0

# Recognition wiring — all None when running in plain camera mode.
_faces_mod = None
_recognizer = None
_gallery = None
_alert_gate = None
_threshold = 0.40
_jobs: "queue.Queue[tuple[str, list]]" = queue.Queue(maxsize=MAX_PENDING)
_dropped = 0

#: Crossings we put a name to, in total and per person. The wall shows both:
#: how many came through, and how many we actually recognised.
_matched = 0
_visits: dict[str, int] = {}

# Alert channels.
WEBHOOK = None
SERIAL_ALERTS = True
_serial = None  # live handle owned by watch_serial(), written to by alerts
_serial_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventBus:
    """Fan out events to SSE subscribers, and keep a short history.

    Subscribers get their own bounded queue; a slow or dead browser tab fills up
    and gets dropped rather than blocking the recognition worker.
    """

    def __init__(self, history: int = HISTORY):
        self._subs: list[queue.Queue] = []
        self._history: deque = deque(maxlen=history)
        self._lock = threading.Lock()
        self._seq = 0

    def publish(self, kind: str, **payload) -> dict:
        with self._lock:
            self._seq += 1
            event = {
                "id": self._seq,
                "kind": kind,
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                **payload,
            }
            self._history.append(event)
            subs = list(self._subs)

        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # that tab is not keeping up; it resyncs on reload
        return event

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)


BUS = EventBus()


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def open_camera(index):
    """Hold the camera open. Opening per-shot costs ~1s and drops the moment."""
    global _cap
    _cap = cv2.VideoCapture(index)
    if not _cap.isOpened():
        raise SystemExit(f"could not open camera {index}")
    # Keep the driver buffer shallow so a burst reads *now*, not from backlog.
    try:
        _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    # Warm up: the first few frames off a webcam are usually dark or green.
    for _ in range(5):
        _cap.read()
    print(f"camera {index} open")


def snap(reason="manual", burst=None):
    """Grab frames and write the first. Returns (path, frames).

    The written JPEG is the first frame — that is the moment the door fired.
    The rest exist only to give recognition more than one shot at the face.
    """
    global _count
    n = BURST if burst is None else max(1, burst)
    frames = []
    with _lock:
        for _ in range(n):
            ok, frame = _cap.read()
            if ok and frame is not None:
                frames.append(frame)

    if not frames:
        print("! camera gave no frame")
        return None, []

    os.makedirs(OUT, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    path = os.path.join(OUT, f"{name}_{reason}.jpg")
    cv2.imwrite(path, frames[0])
    _count += 1
    extra = f" (+{len(frames) - 1} for recognition)" if len(frames) > 1 else ""
    print(f"[{_count}] {path}{extra}")
    return path, frames


def crossing(reason="in"):
    """A person came through: capture, publish, and queue for identification."""
    path, frames = snap(reason, BURST)
    event = BUS.publish(
        "snap", snap=os.path.basename(path) if path else None,
        ok=bool(path), frames=len(frames),
    )
    if path and frames:
        submit(path, frames)
    return path, event


# ---------------------------------------------------------------------------
# Serial — the MCU tells us when, and hears back when we know who
# ---------------------------------------------------------------------------


def watch_serial(port, baud=115200):
    """Snap when the MCU reports an inward passage (`>P I 84`).

    Reconnects on its own — at a venue the board gets unplugged, and the whole
    demo dying because of a loose USB cable is not a good look.

    This also parks the live handle in `_serial` so match alerts can go back out
    over the same link. Opening the tty a second time for writing would fail, so
    read and write share one connection.
    """
    global _serial
    import serial  # only needed on this path

    while True:
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                _serial = ser
                print(f"serial {port} open @ {baud}")
                while True:
                    line = ser.readline().decode("ascii", "replace").strip()
                    if not line:
                        continue
                    # The visitor_counter sketch prints "ENTRY  |  Occupancy: 3 ...".
                    # ">P I" is the older spec'd form — accept both so we work
                    # whichever way the firmware ends up going.
                    if line.startswith("ENTRY") or line.startswith(">P I"):
                        crossing("in")
                    else:
                        print(f"  mcu: {line}")  # eyeball the other events
        except Exception as e:
            print(f"! serial: {e} — retrying in 2s")
            time.sleep(2)
        finally:
            _serial = None


def _serial_alert(event: dict) -> None:
    """Nudge the MCU using the frozen protocol's Linux->MCU sentinels.

    `<K` is an acknowledge tone and `<E <text>` a short display message. A
    7-segment cannot render a name, so the name goes out for the log and the
    tone is the part a human actually notices.

    Both lines are ignorable by a tolerant parser, which the firmware spec
    requires, so if the sketch has not implemented them yet the worst case is
    silence rather than a fault.
    """
    ser = _serial
    if ser is None:
        return
    try:
        short = "".join(c for c in event["name"] if c.isalnum())[:8].upper()
        with _serial_write_lock:
            ser.write(f"<E {short}\n".encode())
            ser.write(b"<K\n")
    except Exception as e:
        print(f"! serial alert failed: {e}")


# ---------------------------------------------------------------------------
# Recognition worker
# ---------------------------------------------------------------------------


def submit(path: str, frames: list) -> None:
    """Hand a crossing to the recogniser without blocking the caller."""
    global _dropped
    if _recognizer is None:
        return
    try:
        _jobs.put_nowait((path, frames))
    except queue.Full:
        _dropped += 1
        print(f"! recognition backlog full — dropped a crossing ({_dropped} total)."
              f" Try --det-size 320, or offload with --recognizer.")
        BUS.publish("dropped", total=_dropped)


def recognition_worker() -> None:
    """Identify crossings off the trigger path, forever."""
    while True:
        path, frames = _jobs.get()
        try:
            t0 = time.perf_counter()
            matches = _faces_mod.identify(
                _recognizer, frames, _gallery, _threshold, want_crops=True
            )
            ms = (time.perf_counter() - t0) * 1000

            if not matches:
                BUS.publish("noface", snap=os.path.basename(path), ms=round(ms))
                print(f"    no face ({ms:.0f}ms)")

            for m in matches:
                if m.matched:
                    _handle_match(m, path, ms)
                else:
                    _handle_unknown(m, path, ms)
        except Exception as e:  # a bad frame must never kill the worker
            print(f"! recognition failed: {e}")
            BUS.publish("error", detail=str(e))
        finally:
            _jobs.task_done()


def _handle_match(m, path: str, ms: float) -> None:
    global _matched
    person = _gallery.people.get(m.name)
    muted = person is not None and not person.alert
    fresh = _alert_gate.should_alert(m.name)

    # Counted per sighting, not per alert: the cooldown governs how loudly we
    # react, not whether the person actually walked through the door.
    _visits[m.name] = _visits.get(m.name, 0) + 1
    _matched += 1

    note = "" if fresh else "  [cooling down]"
    note += "  [muted]" if muted else ""
    print(f"    * {m.name}  score {m.score:.3f}  ({ms:.0f}ms)  "
          f"visit #{_visits[m.name]}{note}")

    event = BUS.publish(
        "match", name=m.name, score=round(m.score, 3), margin=round(m.margin, 3),
        snap=os.path.basename(path), ms=round(ms),
        alerted=bool(fresh and not muted),
        visits=_visits[m.name], matched=_matched, total=_count,
    )
    if fresh and not muted:
        fire_alerts(event)


def _handle_unknown(m, path: str, ms: float) -> None:
    crop = _save_unknown(m, path)
    BUS.publish(
        "unknown", snap=os.path.basename(path), ms=round(ms),
        closest=m.top_name, closest_score=round(m.score, 3), crop=crop,
    )
    print(f"    unknown face ({ms:.0f}ms, closest "
          f"{m.top_name or '-'} {m.score:.2f})")


def fire_alerts(event: dict) -> None:
    """Push a confirmed match out. Channels are isolated: a dead webhook must
    not stop the beep."""
    if WEBHOOK:
        threading.Thread(target=_post_webhook, args=(event,), daemon=True).start()
    if SERIAL_ALERTS:
        _serial_alert(event)


def _post_webhook(event: dict) -> None:
    try:
        req = urllib.request.Request(
            WEBHOOK, data=json.dumps(event).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:
        print(f"! webhook failed: {e}")


def _save_unknown(m, path: str) -> str | None:
    """Keep the crop of anyone we could not place, so they can be enrolled after
    the fact from the UI — the natural way to build a watchlist is to point at
    someone who already walked through."""
    if m.face is None or m.face.crop is None or m.face.crop.size == 0:
        return None
    try:
        os.makedirs(UNKNOWN_DIR, exist_ok=True)
        name = "unknown-" + os.path.basename(path)
        cv2.imwrite(os.path.join(UNKNOWN_DIR, name), m.face.crop)
        return name
    except Exception as e:
        print(f"! could not save unknown crop: {e}")
        return None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _safe_join(directory: str, filename: str) -> str | None:
    """Resolve `filename` inside `directory`, or None if it tries to escape."""
    if not directory:
        return None
    base = os.path.abspath(directory)
    target = os.path.abspath(os.path.join(base, os.path.basename(filename)))
    ok = target.startswith(base + os.sep) and os.path.isfile(target)
    return target if ok else None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        route = urlparse(self.path)
        path = route.path.rstrip("/") or "/"
        if path == "/snap":
            return self._do_snap()
        if path == "/enroll":
            return self._do_enroll(parse_qs(route.query))
        if path == "/reset":
            return self._do_reset()
        self.send_error(404)

    def do_GET(self):
        route = urlparse(self.path)
        path = route.path.rstrip("/") or "/"
        if path == "/":
            return self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
        if path == "/wall":
            return self._send(200, WALL_HTML.encode(), "text/html; charset=utf-8")
        if path == "/status":
            return self._json(self._status())
        if path == "/people":
            return self._json({"people": self._people()})
        if path == "/events":
            return self._sse()
        if path.startswith("/img/snap/"):
            return self._file(OUT, path[len("/img/snap/"):])
        if path.startswith("/img/unknown/"):
            return self._file(UNKNOWN_DIR, path[len("/img/unknown/"):])
        if path.startswith("/img/face/"):
            d = _faces_mod.FACE_IMAGE_DIR if _faces_mod else ""
            return self._file(d, path[len("/img/face/"):])
        self.send_error(404)

    def do_DELETE(self):
        route = urlparse(self.path)
        if route.path.rstrip("/") != "/people":
            return self.send_error(404)
        name = (parse_qs(route.query).get("name") or [""])[0]
        if _gallery is None or not name:
            return self._fail(400, "name required")
        if not _gallery.remove(name):
            return self._fail(404, "not enrolled")
        _gallery.save()
        BUS.publish("gallery", people=self._people())
        self._json({"ok": True})

    # -- handlers -----------------------------------------------------------

    def _do_snap(self):
        path, event = crossing("in")
        self._json({"ok": bool(path), "path": path or "", "event": event["id"]})

    def _do_reset(self):
        """Zero the counters for a fresh run. Enrolments and photos survive —
        you reset between demos, you do not re-enrol between demos."""
        global _count, _matched, _dropped
        _count = 0
        _matched = 0
        _dropped = 0
        _visits.clear()
        if _alert_gate is not None:
            _alert_gate.reset()          # so the next crossing alerts immediately
        BUS.publish("reset")
        print("-- counters reset --")
        self._json({"ok": True})

    def _do_enroll(self, params):
        """Raw image bytes in the body, name in the query string.

        Raw beats multipart here: the browser can `fetch(url, {body: file})`
        directly and we skip a parser that would only ever see one field.
        """
        if _gallery is None:
            return self._fail(503, "recognition is not enabled")
        name = (params.get("name") or [""])[0].strip()
        if not name:
            return self._fail(400, "name required")

        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return self._fail(400, "empty body")
        raw = self.rfile.read(n)

        import numpy as np

        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return self._fail(400, "could not decode image")

        idx = len(_gallery.people[name].images) if name in _gallery.people else 0
        try:
            person, face = _faces_mod.enroll_image(
                _recognizer, _gallery, name, img, _faces_mod.enrolment_path(name, idx)
            )
        except _faces_mod.EnrolmentError as e:
            return self._fail(422, str(e))

        _gallery.save()
        _alert_gate.reset(name)  # a fresh enrolment should alert immediately
        print(f"enrolled {name}: face {face.size_px}px, "
              f"{len(person.embeddings)} reference(s)")
        BUS.publish("gallery", people=self._people(), enrolled=name)
        self._json({"ok": True, "name": name,
                    "references": len(person.embeddings), "face_px": face.size_px})

    def _sse(self):
        q = BUS.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for event in BUS.history():
                self._event(event)
            while True:
                try:
                    self._event(q.get(timeout=15))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")  # keeps idle proxies honest
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            BUS.unsubscribe(q)

    def _event(self, event: dict) -> None:
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()

    # -- helpers ------------------------------------------------------------

    def _status(self) -> dict:
        return {
            "snaps": _count, "dir": OUT, "burst": BURST,
            "recognition": bool(_recognizer),
            "recognizer": _recognizer.name if _recognizer else None,
            "threshold": _threshold,
            "enrolled": len(_gallery) if _gallery is not None else 0,
            "dropped": _dropped, "pending": _jobs.qsize(),
            "matched": _matched, "visits": dict(_visits),
            "webhook": bool(WEBHOOK), "serial": _serial is not None,
        }

    def _people(self) -> list[dict]:
        if _gallery is None:
            return []
        return [{
            "name": p.name, "references": len(p.embeddings), "alert": p.alert,
            "added": p.added,
            "image": os.path.basename(p.images[-1]) if p.images else None,
        } for p in _gallery.people.values()]

    def _file(self, directory: str, filename: str):
        target = _safe_join(directory, filename)
        if not target:
            return self.send_error(404)
        with open(target, "rb") as fh:
            body = fh.read()
        self._send(200, body, "image/jpeg", cache=True)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), "application/json")

    def _fail(self, code: int, message: str):
        """Error with the detail in the body, not the status line.

        The HTTP reason phrase is latin-1 only. Our enrolment messages contain
        an em dash, which raises UnicodeEncodeError *while writing the
        response* — the browser sees a dropped connection instead of "no face
        found", exactly when someone is trying to enrol a bad photo. The body
        has no such limit, and the UI already reads it.
        """
        self._send(code, message.encode("utf-8"), "text/plain; charset=utf-8")

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        elif ctype.startswith("text/html"):
            # Without this the browser heuristically caches the page and you
            # end up debugging a build you are no longer running.
            self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def handle_one_request(self):
        # A browser tab closing an SSE stream resets the socket, and the default
        # handler dumps a full traceback for it. Expected, not exceptional —
        # and a console full of stack traces during a demo reads as broken.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def log_message(self, *a):
        pass  # the snap and match lines are the only output worth having


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TallyPoint - faces</title>
<style>
  :root{--bg:#0e1116;--panel:#161b22;--line:#262c36;--fg:#e6edf3;--dim:#8b949e;
        --hit:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif}
  header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
         padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
  h1{margin:0;font-size:16px;letter-spacing:.02em}
  .meta{color:var(--dim);font-size:12px}
  .dot{display:inline-block;width:7px;height:7px;border-radius:50%;
       background:var(--dim);margin-right:5px;vertical-align:middle}
  main{display:grid;grid-template-columns:340px 1fr;gap:18px;padding:18px;
       max-width:1200px;margin:0 auto;align-items:start}
  @media(max-width:820px){main{grid-template-columns:1fr}}
  section{background:var(--panel);border:1px solid var(--line);border-radius:10px;
          padding:14px}
  h2{margin:0 0 10px;font-size:12px;text-transform:uppercase;letter-spacing:.08em;
     color:var(--dim);font-weight:600}
  input[type=text]{width:100%;padding:8px 10px;border-radius:7px;
    border:1px solid var(--line);background:#0d1117;color:var(--fg);font:inherit}
  #drop{margin-top:9px;border:1.5px dashed var(--line);border-radius:9px;
    padding:22px 12px;text-align:center;color:var(--dim);cursor:pointer;
    transition:.15s;font-size:13px}
  #drop:hover,#drop.over{border-color:var(--accent);color:var(--accent);
    background:rgba(88,166,255,.06)}
  #msg{margin-top:9px;font-size:12.5px;min-height:1.2em}
  .ok{color:var(--hit)} .err{color:var(--bad)}
  ul{list-style:none;margin:0;padding:0}
  .person{display:flex;align-items:center;gap:10px;padding:7px 0;
    border-top:1px solid var(--line)}
  .person:first-child{border-top:0}
  .person img{width:38px;height:38px;border-radius:7px;object-fit:cover;
    background:#0d1117;flex:0 0 auto}
  .person .n{font-weight:600}
  .person .r{color:var(--dim);font-size:12px}
  .person button{margin-left:auto;background:none;border:0;color:var(--dim);
    cursor:pointer;font-size:16px;padding:2px 6px;border-radius:5px}
  .person button:hover{color:var(--bad);background:rgba(248,81,73,.1)}
  #feed{display:flex;flex-direction:column;gap:8px;max-height:74vh;overflow:auto}
  .ev{display:flex;align-items:center;gap:11px;padding:9px 11px;border-radius:9px;
    background:#0d1117;border:1px solid var(--line);border-left-width:3px}
  .ev.match{border-left-color:var(--hit)}
  .ev.unknown{border-left-color:var(--warn)}
  .ev.noface,.ev.snap{border-left-color:var(--line)}
  .ev.error,.ev.dropped{border-left-color:var(--bad)}
  .ev img{width:46px;height:46px;border-radius:6px;object-fit:cover;flex:0 0 auto}
  .ev .who{font-weight:600} .ev .who.g{color:var(--hit)} .ev .who.w{color:var(--warn)}
  .ev .sub{color:var(--dim);font-size:12px}
  .ev time{margin-left:auto;color:var(--dim);font-size:11.5px;
    font-variant-numeric:tabular-nums;white-space:nowrap}
  .empty{color:var(--dim);text-align:center;padding:26px 0;font-size:13px}
  button.enrol{margin-left:auto;background:rgba(88,166,255,.14);border:0;
    color:var(--accent);cursor:pointer;font-size:12px;padding:4px 9px;border-radius:6px}
</style></head><body>
<header>
  <h1>TallyPoint</h1>
  <span class="meta"><span class="dot" id="live"></span><span id="stat">connecting...</span></span>
  <a href="/wall" target="_blank" style="margin-left:auto;color:var(--accent);
     text-decoration:none;font-size:12.5px;letter-spacing:.06em">open the wall →</a>
</header>
<main>
  <div>
    <section>
      <h2>Enrol a face</h2>
      <input type="text" id="name" placeholder="Name" autocomplete="off">
      <div id="drop">Drop a photo, or click to choose<br><small>front-on, well lit</small></div>
      <input type="file" id="file" accept="image/*" hidden multiple>
      <div id="msg"></div>
    </section>
    <section style="margin-top:16px">
      <h2>Watchlist</h2>
      <ul id="people"><li class="empty">No one enrolled yet</li></ul>
    </section>
  </div>
  <section>
    <h2>Live</h2>
    <button id="enter" style="width:100%;margin-bottom:12px;padding:13px;border:0;
      border-radius:9px;background:var(--accent);color:#04121f;font:inherit;
      font-weight:800;letter-spacing:.14em;text-transform:uppercase;cursor:pointer">
      Person enters</button>
    <button id="dreset" style="width:100%;margin-bottom:12px;padding:9px;
      border:1px solid var(--line);border-radius:9px;background:none;color:var(--dim);
      font:inherit;font-size:12px;letter-spacing:.14em;text-transform:uppercase;
      cursor:pointer">Reset counters</button>
    <div id="feed"><div class="empty">Waiting for a crossing...</div></div>
  </section>
</main>
<script>
const $=s=>document.querySelector(s), feed=$('#feed');
let people=[];
const esc=s=>String(s??'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
const fmt=ts=>new Date(ts).toLocaleTimeString();

function row(e){
  const d=document.createElement('div'); d.className='ev '+e.kind;
  let img='',who='',sub='';
  if(e.kind==='match'){
    img=e.snap?`/img/snap/${e.snap}`:'';
    who=`<span class="who g">${esc(e.name)}</span>`;
    sub=`match ${e.score} · margin ${e.margin} · ${e.ms}ms`
       +(e.alerted?' · <b>alerted</b>':' · cooling down');
  } else if(e.kind==='unknown'){
    img=e.crop?`/img/unknown/${e.crop}`:'';
    who='<span class="who w">Unknown</span>';
    sub=`closest ${esc(e.closest)||'-'} ${e.closest_score??''} · ${e.ms}ms`;
  } else if(e.kind==='noface'){
    img=e.snap?`/img/snap/${e.snap}`:''; who='No face'; sub=`crossing · ${e.ms}ms`;
  } else if(e.kind==='snap'){
    img=e.snap?`/img/snap/${e.snap}`:''; who='Crossing';
    sub=`${e.frames} frame(s) captured`;
  } else if(e.kind==='dropped'){
    who='Dropped'; sub=`recogniser fell behind (${e.total} total)`;
  } else if(e.kind==='error'){ who='Error'; sub=esc(e.detail); }
  else return null;

  d.innerHTML=(img?`<img src="${img}" loading="lazy" onerror="this.remove()">`:'')
    +`<div><div>${who}</div><div class="sub">${sub}</div></div>`
    +`<time>${fmt(e.ts)}</time>`;

  if(e.kind==='unknown'&&e.crop){
    const b=document.createElement('button'); b.className='enrol'; b.textContent='Enrol';
    b.onclick=async()=>{
      const n=prompt('Name for this person?'); if(!n)return;
      const blob=await (await fetch(`/img/unknown/${e.crop}`)).blob();
      upload(n,new File([blob],'crop.jpg',{type:'image/jpeg'}));
    };
    d.appendChild(b);
  }
  return d;
}

function push(e){
  const r=row(e); if(!r)return;
  const empty=feed.querySelector('.empty'); if(empty)feed.innerHTML='';
  feed.prepend(r);
  while(feed.children.length>80)feed.lastChild.remove();
}

function renderPeople(){
  const ul=$('#people');
  if(!people.length){ul.innerHTML='<li class="empty">No one enrolled yet</li>';return;}
  ul.innerHTML='';
  for(const p of people){
    const li=document.createElement('li'); li.className='person';
    li.innerHTML=(p.image?`<img src="/img/face/${p.image}" onerror="this.remove()">`:'<img>')
      +`<div><div class="n">${esc(p.name)}</div>`
      +`<div class="r">${p.references} reference${p.references===1?'':'s'}</div></div>`;
    const b=document.createElement('button'); b.textContent='x'; b.title='Remove';
    b.onclick=async()=>{
      if(!confirm(`Remove ${p.name}?`))return;
      await fetch('/people?name='+encodeURIComponent(p.name),{method:'DELETE'});
    };
    li.appendChild(b); ul.appendChild(li);
  }
}

async function upload(name,file){
  const m=$('#msg'); m.className=''; m.textContent=`Enrolling ${name}...`;
  try{
    const r=await fetch('/enroll?name='+encodeURIComponent(name),
      {method:'POST',body:file,headers:{'Content-Type':file.type||'image/jpeg'}});
    if(!r.ok){m.className='err';m.textContent=(await r.text())||`Failed (${r.status})`;return;}
    const j=await r.json();
    m.className='ok';
    m.textContent=`OK ${j.name} - ${j.references} reference${j.references===1?'':'s'}, face ${j.face_px}px`;
  }catch(err){m.className='err';m.textContent=String(err);}
}

function pick(files){
  const name=$('#name').value.trim();
  if(!name){$('#msg').className='err';$('#msg').textContent='Enter a name first';return;}
  for(const f of files) upload(name,f);
}
$('#drop').onclick=()=>$('#file').click();
$('#file').onchange=e=>pick(e.target.files);
$('#drop').ondragover=e=>{e.preventDefault();$('#drop').classList.add('over')};
$('#drop').ondragleave=()=>$('#drop').classList.remove('over');
$('#drop').ondrop=e=>{e.preventDefault();$('#drop').classList.remove('over');
                      pick(e.dataTransfer.files)};

async function refresh(){
  try{
    const s=await (await fetch('/status')).json();
    $('#stat').textContent=s.recognition
      ? `${s.recognizer} · thresh ${s.threshold} · ${s.enrolled} enrolled · `
        +`${s.snaps} snaps · burst ${s.burst}`+(s.serial?' · serial':'')
      : `camera only - recognition disabled · ${s.snaps} snaps`;
    people=(await (await fetch('/people')).json()).people; renderPeople();
  }catch(e){}
}

$('#dreset').onclick=async ()=>{ await fetch('/reset',{method:'POST'}); feed.innerHTML=''; };

$('#enter').onclick=async ev=>{
  const b=ev.target; b.disabled=true; b.style.opacity=.5;
  try{ await fetch('/snap',{method:'POST'}); }catch(e){}
  setTimeout(()=>{b.disabled=false;b.style.opacity=1;},1200);
};

const es=new EventSource('/events');
es.onmessage=m=>{
  const e=JSON.parse(m.data);
  if(e.kind==='gallery'){people=e.people;renderPeople();refresh();}
  else push(e);
};
es.onerror=()=>{$('#live').style.background='var(--bad)';
                $('#stat').textContent='disconnected - is snap.py running?'};
es.onopen=()=>{$('#live').style.background='var(--hit)';refresh()};
refresh();
</script></body></html>
"""


# ---------------------------------------------------------------------------
# The wall — the page you put on the projector
# ---------------------------------------------------------------------------
#
# Separate from the dashboard on purpose. `/` is for operating the thing: small
# type, dense, everything visible at once. `/wall` is for an audience across a
# room: one thing at a time, enormous, with a beat of suspense while the model
# thinks. That pause is real — it is the recogniser working — so the theatre is
# honest rather than staged.

WALL_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TallyPoint - the wall</title>
<style>
  :root{--bg:#08090d;--fg:#f2f6fc;--dim:#6f7787;--hit:#3fb950;--warn:#f0a92b;
        --accent:#58a6ff}
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%;overflow:hidden}
  body{background:var(--bg);color:var(--fg);
    font:16px/1.4 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif}
  body.idle{cursor:none}
  body.idle #trigger{opacity:.12}
  /* The demo's only control. Discreet so it does not compete with the reveal,
     but the spacebar is the one you actually want on stage. */
  #trigger{position:fixed;bottom:2.4vh;left:50%;transform:translateX(-50%);
    z-index:70;border:0;border-radius:99px;cursor:pointer;transition:opacity .4s;
    padding:14px 30px;font:inherit;font-weight:800;letter-spacing:.16em;
    text-transform:uppercase;font-size:clamp(11px,1.15vw,16px);
    background:var(--accent);color:#04121f;box-shadow:0 10px 34px rgba(88,166,255,.3)}
  #trigger:hover{opacity:1!important;transform:translateX(-50%) scale(1.04)}
  #trigger:active{transform:translateX(-50%) scale(.97)}
  #trigger:disabled{background:#20262f;color:var(--dim);box-shadow:none;cursor:default}
  canvas#confetti{position:fixed;inset:0;pointer-events:none;z-index:60}

  .stage{position:fixed;inset:0;display:grid;place-items:center;text-align:center}
  .layer{position:absolute;inset:0;display:grid;place-items:center;
    opacity:0;pointer-events:none;transition:opacity .25s}
  .layer.on{opacity:1}

  /* ---- idle ---- */
  #idle .eye{width:min(26vh,240px);aspect-ratio:1;border-radius:50%;
    border:3px solid #1b2029;position:relative;display:grid;place-items:center;
    margin:0 auto 5vh}
  #idle .eye::after{content:"";position:absolute;inset:-3px;border-radius:50%;
    border:3px solid transparent;border-top-color:var(--accent);
    animation:spin 2.4s linear infinite}
  #idle .pupil{width:34%;aspect-ratio:1;border-radius:50%;background:#151a22;
    animation:breathe 3s ease-in-out infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes breathe{0%,100%{transform:scale(1);opacity:.7}50%{transform:scale(1.14);opacity:1}}
  #idle h1{font-size:clamp(22px,3.4vw,52px);font-weight:800;letter-spacing:.16em;
    color:var(--dim)}
  #idle p{margin-top:2.2vh;color:#3d4450;font-size:clamp(13px,1.3vw,20px);
    letter-spacing:.22em;text-transform:uppercase}

  /* ---- scanning ---- */
  #scan .frame{position:relative;width:min(48vh,44vw);aspect-ratio:4/3;
    border-radius:18px;overflow:hidden;background:#0d1117;
    box-shadow:0 0 0 3px #1b2029,0 30px 80px rgba(0,0,0,.6)}
  #scan img{width:100%;height:100%;object-fit:cover;filter:saturate(.4) contrast(1.1)}
  #scan .beam{position:absolute;left:0;right:0;height:34%;
    background:linear-gradient(180deg,transparent,rgba(88,166,255,.28),transparent);
    animation:sweep 1.05s cubic-bezier(.5,0,.5,1) infinite}
  @keyframes sweep{0%{top:-34%}100%{top:100%}}
  #scan h2{margin-top:4vh;font-size:clamp(18px,2.4vw,38px);letter-spacing:.3em;
    color:var(--accent);font-weight:700;text-transform:uppercase}
  #scan .dots::after{content:"";animation:dots 1.2s steps(4,end) infinite}
  @keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}}

  /* ---- reveal ---- */
  #hit .pair{display:flex;align-items:center;gap:clamp(16px,3vw,56px)}
  #hit figure{display:grid;gap:1.4vh;justify-items:center}
  #hit img{width:clamp(120px,25vh,260px);aspect-ratio:1;object-fit:cover;
    border-radius:22px;background:#0d1117;box-shadow:0 22px 60px rgba(0,0,0,.55)}
  #hit figcaption{font-size:clamp(10px,.95vw,14px);letter-spacing:.24em;
    text-transform:uppercase;color:var(--dim)}
  #hit .link{display:grid;gap:.8vh;justify-items:center}
  #hit .eq{font-size:clamp(26px,4vw,60px);font-weight:800;color:var(--hit);
    animation:thump 1.1s ease-in-out infinite}
  @keyframes thump{0%,100%{transform:scale(1)}50%{transform:scale(1.16)}}
  #hit .pct{font-variant-numeric:tabular-nums;font-size:clamp(13px,1.4vw,22px);
    color:var(--hit);font-weight:700;letter-spacing:.06em}
  #hit .name{margin-top:4.5vh;font-size:clamp(38px,9vw,150px);font-weight:900;
    letter-spacing:-.02em;line-height:.95;
    background:linear-gradient(180deg,#fff,#8fd3a3);
    -webkit-background-clip:text;background-clip:text;color:transparent}
  #hit .quip{margin-top:2vh;font-size:clamp(14px,1.9vw,32px);color:var(--hit);
    letter-spacing:.2em;text-transform:uppercase;font-weight:700}
  #hit .meta{margin-top:2.4vh;color:var(--dim);font-size:clamp(11px,1.1vw,17px);
    letter-spacing:.13em;text-transform:uppercase}

  .pop{animation:pop .62s cubic-bezier(.18,1.5,.4,1) both}
  @keyframes pop{from{transform:scale(.62);opacity:0}to{transform:scale(1);opacity:1}}
  .rise{animation:rise .5s cubic-bezier(.2,.9,.3,1) both}
  @keyframes rise{from{transform:translateY(26px);opacity:0}to{transform:none;opacity:1}}
  .d1{animation-delay:.12s} .d2{animation-delay:.24s} .d3{animation-delay:.36s}

  /* ---- unknown ---- */
  #miss img{width:clamp(120px,26vh,270px);aspect-ratio:1;object-fit:cover;
    border-radius:22px;background:#0d1117;filter:grayscale(.6);
    box-shadow:0 22px 60px rgba(0,0,0,.55)}
  #miss .q{margin-top:4vh;font-size:clamp(34px,8vw,130px);font-weight:900;
    color:var(--warn);letter-spacing:-.01em;line-height:.95}
  #miss .quip{margin-top:2vh;font-size:clamp(13px,1.7vw,28px);color:#8a6d29;
    letter-spacing:.2em;text-transform:uppercase;font-weight:700}

  /* ---- chrome ---- */
  #count{position:fixed;top:3vh;right:3.2vw;z-index:20;display:flex;
    gap:clamp(18px,2.6vw,44px);text-align:right}
  #count .c b{display:block;font-size:clamp(30px,4.8vw,76px);font-weight:900;
    line-height:1;font-variant-numeric:tabular-nums}
  #count .c span{font-size:clamp(9px,.9vw,13px);letter-spacing:.24em;
    color:var(--dim);text-transform:uppercase}
  #count .c.hit b{color:var(--hit)}
  .bump{animation:bump .5s cubic-bezier(.2,1.6,.4,1)}
  @keyframes bump{0%{transform:scale(1)}40%{transform:scale(1.32)}100%{transform:scale(1)}}

  /* Alarm flash — pulses the screen edges red in time with the klaxon. */
  #flash{position:fixed;inset:0;z-index:50;pointer-events:none;opacity:0;
    box-shadow:inset 0 0 min(24vh,220px) min(4vh,50px) rgba(248,81,73,.9)}
  #flash.go{animation:flash 1.15s ease-out}
  @keyframes flash{0%{opacity:0}8%{opacity:1}25%{opacity:.15}42%{opacity:1}
    60%{opacity:.15}76%{opacity:.9}100%{opacity:0}}

  #hit .visit{display:inline-block;margin-top:2vh;padding:7px 20px;border-radius:99px;
    background:rgba(63,185,80,.16);color:var(--hit);font-weight:800;
    font-size:clamp(12px,1.4vw,22px);letter-spacing:.18em;text-transform:uppercase}
  #ticker{position:fixed;bottom:0;left:0;right:0;display:flex;gap:10px;
    padding:1.6vh 3.2vw;justify-content:center;z-index:20}
  .chip{font-size:clamp(10px,.95vw,14px);padding:6px 13px;border-radius:99px;
    background:#11151c;border:1px solid #1b2029;color:var(--dim);white-space:nowrap;
    letter-spacing:.06em}
  .chip.m{color:var(--hit);border-color:#1d3a26;background:#0f1b13}
  .chip.u{color:var(--warn);border-color:#3a2f16;background:#1b1710}
  #sound{position:fixed;bottom:0;left:0;right:0;top:0;z-index:90;cursor:pointer;
    display:grid;place-items:center;background:rgba(8,9,13,.94);
    font-size:clamp(15px,2.1vw,30px);letter-spacing:.2em;text-transform:uppercase;
    color:var(--dim)}
  #reset{position:fixed;bottom:2.6vh;right:3.2vw;z-index:70;cursor:pointer;
    background:none;border:1px solid var(--line);color:var(--dim);border-radius:99px;
    padding:8px 16px;font:inherit;font-size:clamp(10px,.95vw,13px);letter-spacing:.16em;
    text-transform:uppercase;transition:opacity .4s}
  #reset:hover{color:var(--bad);border-color:var(--bad)}
  body.idle #reset{opacity:.1}
  #build{position:fixed;bottom:2.9vh;left:3.2vw;z-index:70;color:#2a303a;
    font-size:11px;letter-spacing:.14em;text-transform:uppercase}
  #dead{position:fixed;top:3vh;left:3.2vw;color:var(--warn);z-index:20;
    font-size:clamp(10px,1vw,15px);letter-spacing:.16em;text-transform:uppercase;
    display:none}
</style></head><body>

<div id="sound">Click anywhere to arm the sound</div>
<div id="dead">◉ disconnected</div>
<button id="trigger">Person enters &nbsp;·&nbsp; space</button>
<button id="reset" title="Zero the counters">reset</button>
<div id="build">build 5</div>
<div id="ticker"></div>
<div id="count">
  <div class="c"><b id="ctotal">0</b><span>through the door</span></div>
  <div class="c hit"><b id="cmatch">0</b><span>recognised</span></div>
</div>
<div id="flash"></div>
<canvas id="confetti"></canvas>

<div class="stage">
  <div class="layer on" id="idle">
    <div>
      <div class="eye"><div class="pupil"></div></div>
      <h1>WATCHING THE DOOR</h1>
      <p id="idlesub">nobody yet</p>
    </div>
  </div>

  <div class="layer" id="scan">
    <div>
      <div class="frame"><img id="scanimg" alt=""><div class="beam"></div></div>
      <h2>Who goes there<span class="dots"></span></h2>
    </div>
  </div>

  <div class="layer" id="hit">
    <div>
      <div class="pair">
        <figure class="pop"><img id="hitref" alt=""><figcaption>on file</figcaption></figure>
        <div class="link pop d1">
          <div class="eq">=</div><div class="pct" id="hitpct">0%</div>
        </div>
        <figure class="pop d2"><img id="hitlive" alt=""><figcaption>just now</figcaption></figure>
      </div>
      <div class="name rise d2" id="hitname">—</div>
      <div class="quip rise d3" id="hitquip"></div>
      <div class="visit rise d3" id="hitvisit"></div>
      <div class="meta rise d3" id="hitmeta"></div>
    </div>
  </div>

  <div class="layer" id="miss">
    <div>
      <img id="missimg" class="pop" alt="">
      <div class="q rise d1">WHO IS THIS</div>
      <div class="quip rise d2" id="missquip"></div>
    </div>
  </div>
</div>

<script>
const $=s=>document.querySelector(s);
const HOLD=4200, SCAN_MIN=550;   // reveal dwell, and a floor on the suspense beat

const QUIPS=["has entered the chat","in the building","the legend arrives",
  "we've been expecting you","look who showed up","certified doorway royalty",
  "somebody get them a chair","long time no see (4 seconds)","back again, we see"];
const MISSQUIPS=["new challenger approaching","not on the list, buddy",
  "stranger danger (probably fine)","identify yourself","a wild human appears",
  "no idea. none whatsoever."];
const pick=a=>a[Math.floor(Math.random()*a.length)];

// An explicit state machine. The previous version used one `busy` flag for
// both "a scan is on screen" and "a reveal is on screen", so a verdict arriving
// mid-scan queued behind its own scan instead of resolving it.
let people={}, crossings=0, pending=[], scanShownAt=0;
let state='idle';                      // 'idle' | 'scan' | 'reveal'
let scanTimer=null, holdTimer=null;

function show(id){
  for(const l of document.querySelectorAll('.layer')) l.classList.toggle('on',l.id===id);
}
function replay(sel){  // restart CSS animations on re-show
  for(const el of document.querySelectorAll(sel)){
    el.style.animation='none'; void el.offsetHeight; el.style.animation='';
  }
}

/* ---------- sound: a small fanfare, synthesised so there is no asset to ship ---- */
let AC=null;
function arm(){ if(!AC){AC=new (window.AudioContext||window.webkitAudioContext)();}
                if(AC.state==='suspended')AC.resume(); $('#sound').style.display='none'; }
function tone(freq,at,dur,type='triangle',gain=.22){
  if(!AC)return;
  const o=AC.createOscillator(), g=AC.createGain();
  o.type=type; o.frequency.setValueAtTime(freq,AC.currentTime+at);
  g.gain.setValueAtTime(0,AC.currentTime+at);
  g.gain.linearRampToValueAtTime(gain,AC.currentTime+at+.012);
  g.gain.exponentialRampToValueAtTime(.0001,AC.currentTime+at+dur);
  o.connect(g).connect(AC.destination); o.start(AC.currentTime+at); o.stop(AC.currentTime+at+dur+.02);
}
// A two-tone klaxon, not a chime. It has to read as "alert" from the back of
// a room, and it has to cut through people talking.
function klaxon(){
  for(let i=0;i<3;i++){
    const t=i*.34;
    tone(740,t,.19,'square',.30);
    tone(560,t+.17,.19,'square',.30);
  }
}
const fanfare=()=>[[523,.95],[659,1.04],[784,1.13],[1047,1.22]]
  .forEach(([f,t])=>tone(f,t,.30,'triangle',.17));
const womp=()=>{tone(320,0,.22,'sawtooth',.16);tone(240,.18,.34,'sawtooth',.16);};
const blip=()=>tone(880,0,.05,'square',.07);

/* ---------- confetti: hand-rolled, ~40 lines, no library ---------- */
const cvs=$('#confetti'), ctx=cvs.getContext('2d');
let bits=[];
function fit(){
  const d=Math.min(devicePixelRatio||1,2);
  cvs.width=innerWidth*d; cvs.height=innerHeight*d;
  cvs.style.width=innerWidth+'px'; cvs.style.height=innerHeight+'px';
  ctx.setTransform(d,0,0,d,0,0);
}
addEventListener('resize',fit); fit();

const CONF=['#3fb950','#58a6ff','#f0a92b','#ff7b72','#d2a8ff','#ffffff','#ffd93d'];

function spawn(x,y,vx,vy,n,spread){
  for(let i=0;i<n;i++){
    const a=Math.atan2(vy,vx)+(Math.random()-.5)*spread;
    const sp=Math.hypot(vx,vy)*(.55+Math.random()*.85);
    const kind=Math.random();
    bits.push({
      x,y, vx:Math.cos(a)*sp, vy:Math.sin(a)*sp,
      // streamers flutter and hang; discs and squares fall faster
      shape: kind<.18?'streamer':(kind<.5?'circle':'rect'),
      w:5+Math.random()*9, h:4+Math.random()*8,
      c:CONF[(Math.random()*CONF.length)|0],
      r:Math.random()*Math.PI*2, vr:(Math.random()-.5)*.42,
      // phase drives the flutter: the piece turns edge-on and catches the light
      ph:Math.random()*Math.PI*2, vph:.12+Math.random()*.22,
      drag:.985+Math.random()*.012, life:1, fade:.008+Math.random()*.012
    });
  }
}

function burst(){
  const W=innerWidth,H=innerHeight;
  // Two side cannons firing inward and up — reads much richer than one plume.
  spawn(-10,        H*.72,  22, -19, 90, .55);
  spawn(W+10,       H*.72, -22, -19, 90, .55);
  // Centre fountain, for the beat right under the name.
  spawn(W*.5, H*.60,   0, -25, 80, 1.5);
  if(bits.length>900) bits.splice(0,bits.length-900);   // stay smooth
}

(function tick(){
  ctx.clearRect(0,0,innerWidth,innerHeight);
  bits=bits.filter(b=>b.life>0 && b.y<innerHeight+60);
  for(const b of bits){
    b.vy+=.34;                       // gravity
    b.vx*=b.drag; b.vy*=b.drag;      // air
    b.x+=b.vx; b.y+=b.vy;
    b.r+=b.vr; b.ph+=b.vph;
    if(b.y>innerHeight*.5) b.life-=b.fade;

    const flutter=Math.cos(b.ph);    // -1..1, the piece turning over
    ctx.save();
    ctx.globalAlpha=Math.max(0,Math.min(1,b.life));
    ctx.translate(b.x,b.y); ctx.rotate(b.r);
    ctx.scale(1, Math.max(.12,Math.abs(flutter)));   // edge-on foreshortening
    ctx.fillStyle=b.c;
    // Dim the reverse side so the flutter actually reads as a turning piece.
    if(flutter<0) ctx.globalAlpha*=.55;
    if(b.shape==='circle'){
      ctx.beginPath(); ctx.arc(0,0,b.w*.45,0,6.284); ctx.fill();
    } else if(b.shape==='streamer'){
      ctx.fillRect(-b.w*.22,-b.h*1.5,b.w*.44,b.h*3);
    } else {
      ctx.fillRect(-b.w/2,-b.h/2,b.w,b.h);
    }
    ctx.restore();
  }
  requestAnimationFrame(tick);
})();

/* ---------- the show ---------- */
function alarm(){
  klaxon();
  const f=$('#flash'); f.classList.remove('go'); void f.offsetWidth; f.classList.add('go');
}

function bumpCounters(e){
  if(e.total!==undefined){crossings=e.total;$('#ctotal').textContent=e.total;}
  if(e.matched!==undefined){
    const el=$('#cmatch'); el.textContent=e.matched;
    el.classList.remove('bump'); void el.offsetWidth; el.classList.add('bump');
  }
}

function idle(){
  state='idle'; clearTimeout(scanTimer); clearTimeout(holdTimer); show('idle');
  $('#idlesub').textContent = crossings ? `${crossings} through so far` : 'nobody yet';
  if(pending.length) setTimeout(()=>run(pending.shift()),120);
}

function run(e){
  state='reveal'; clearTimeout(holdTimer);
  if(e.kind==='match'){
    const p=people[e.name]||{};
    $('#hitref').src = p.image?`/img/face/${p.image}`:'';
    $('#hitlive').src= e.snap?`/img/snap/${e.snap}`:'';
    $('#hitname').textContent=e.name.toUpperCase();
    $('#hitquip').textContent=pick(QUIPS);
    $('#hitmeta').textContent=`${e.ms} ms · margin ${e.margin}`
      +(e.alerted?'':' · already greeted');
    let pct=0; const target=Math.round(e.score*100);
    const iv=setInterval(()=>{pct=Math.min(target,pct+Math.ceil(target/18));
      $('#hitpct').textContent=pct+'%'; if(pct>=target)clearInterval(iv);},26);
    $('#hitvisit').textContent = e.visits ? `visit #${e.visits}` : '';
    bumpCounters(e);
    replay('#hit .pop,#hit .rise'); show('hit');
    alarm(); fanfare(); burst();          // klaxon first, resolve into the fanfare
    holdTimer=setTimeout(idle,HOLD);
  } else if(e.kind==='unknown'){
    $('#missimg').src=e.crop?`/img/unknown/${e.crop}`:'';
    $('#missquip').textContent=pick(MISSQUIPS);
    replay('#miss .pop,#miss .rise'); show('miss'); womp();
    holdTimer=setTimeout(idle,HOLD-1400);
  } else { idle(); }
}

// A crossing arrives before the verdict does. Show the scan immediately — the
// pause that follows is the model actually working, which is the best part.
function scanning(e){
  if(state==='reveal') return;         // a reveal owns the screen; let it finish
  state='scan'; scanShownAt=Date.now();
  $('#scanimg').src=e.snap?`/img/snap/${e.snap}`:'';
  show('scan'); blip();
  // Only a safety net: if the verdict never lands, do not strand the scan.
  clearTimeout(scanTimer);
  scanTimer=setTimeout(()=>{ if(state==='scan') idle(); }, 3000);
}

function verdict(e){
  // A verdict for the scan on screen resolves it now. Only a verdict landing
  // during someone else's reveal has to wait.
  if(state==='reveal'){
    pending.push(e);
    while(pending.length>1) pending.shift();   // stay in sync with the door
    return;
  }
  clearTimeout(scanTimer);
  const wait=Math.max(0,SCAN_MIN-(Date.now()-scanShownAt));
  setTimeout(()=>run(e),wait);
}

function chip(e){
  const t=$('#ticker'); if(!t) return;
  const d=document.createElement('div');
  if(e.kind==='match'){d.className='chip m';d.textContent=`${e.name} ${e.score}`;}
  else if(e.kind==='unknown'){d.className='chip u';d.textContent='unknown';}
  else if(e.kind==='noface'){d.className='chip';d.textContent='no face';}
  else return;
  t.prepend(d); while(t.children.length>7)t.lastChild.remove();
}

async function loadPeople(){
  try{
    const r=await (await fetch('/people')).json();
    people={}; for(const p of r.people) people[p.name]=p;
  }catch(e){}
}
async function loadCount(){
  try{ const s=await (await fetch('/status')).json();
       crossings=s.snaps; $('#ctotal').textContent=crossings;
       $('#cmatch').textContent=s.matched||0; }catch(e){}
}

const es=new EventSource('/events');
let primed=false;
es.onopen=()=>{$('#dead').style.display='none';loadPeople();loadCount();
               setTimeout(()=>primed=true,400)};  // ignore the replayed history
es.onerror=()=>{$('#dead').style.display='block'};
es.onmessage=m=>{
  try{ dispatch(JSON.parse(m.data)); }
  catch(err){ console.error('event handler:',err); }   // never stall the wall
};

function dispatch(e){
  if(e.kind==='gallery'){people={};for(const p of e.people)people[p.name]=p;return;}
  if(e.kind==='reset'){
    crossings=0; pending=[];
    $('#ctotal').textContent='0'; $('#cmatch').textContent='0';
    const t=$('#ticker'); if(t) t.innerHTML=''; idle(); return;
  }
  if(!primed) return;
  if(e.kind==='snap'){crossings++;$('#ctotal').textContent=crossings;scanning(e);return;}
  chip(e);
  if(e.kind==='match'||e.kind==='unknown'||e.kind==='noface'){ verdict(e); }
}

/* ---------- the trigger: this is the demo ---------- */
const btn=$('#trigger');
let firing=false;
async function enter(){
  if(firing) return;                     // double-tap would queue two crossings
  firing=true; btn.disabled=true; arm();
  try{ await fetch('/snap',{method:'POST'}); }
  catch(e){ $('#stat')&&0; }
  // Re-arm once the reveal has had its moment, so the button tracks the show.
  setTimeout(()=>{firing=false;btn.disabled=false;},1500);
}
btn.onclick=e=>{e.stopPropagation();enter()};
$('#reset').onclick=async e=>{
  e.stopPropagation();
  await fetch('/reset',{method:'POST'});
};
addEventListener('keydown',e=>{
  if(e.code==='Space'||e.key===' '){e.preventDefault();enter();}
});

/* Hide the cursor when it stops moving, so a mouse parked on screen does not
   sit in the middle of the projection. */
let idleTimer;
function stir(){
  document.body.classList.remove('idle');
  clearTimeout(idleTimer);
  idleTimer=setTimeout(()=>document.body.classList.add('idle'),2500);
}
addEventListener('mousemove',stir); stir();

addEventListener('click',arm); addEventListener('keydown',arm);
loadPeople(); loadCount();
</script></body></html>
"""


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def setup_recognition(args) -> bool:
    """Wire up the face pipeline. Returns False to fall back to camera-only."""
    global _faces_mod, _recognizer, _gallery, _alert_gate, _threshold

    try:
        import faces as faces_mod
    except ImportError as e:
        print(f"! faces.py not importable ({e}) — camera only")
        return False

    _faces_mod = faces_mod
    _threshold = args.threshold
    _gallery = faces_mod.Gallery.load(args.gallery, args.model)

    if _gallery.people and _gallery.model != args.model:
        print(f"! gallery was built with '{_gallery.model}' but --model is "
              f"'{args.model}'.\n  Embeddings from different packs are not "
              f"comparable. Use --model {_gallery.model}, or re-enrol.")
        return False

    _recognizer = faces_mod.make_recognizer(
        args.recognizer, args.model, args.det_size, threads=args.threads
    )
    _alert_gate = faces_mod.AlertGate(args.cooldown)

    try:
        print("warming recogniser...")
        t0 = time.time()
        _recognizer.warm()
        print(f"recogniser ready in {time.time() - t0:.1f}s ({_recognizer.name})")
    except Exception as e:
        print(f"! could not start recogniser: {e}")
        print("  continuing with camera only — fix and restart to enable matching")
        _recognizer = None
        return False

    threading.Thread(target=recognition_worker, daemon=True).start()
    print(f"recognition on — {len(_gallery)} enrolled, threshold {_threshold}, "
          f"cooldown {args.cooldown:.0f}s")
    return True


def main():
    global OUT, CAM, BURST, WEBHOOK, SERIAL_ALERTS

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--out", default="data/snaps")
    ap.add_argument("--serial", help="MCU serial port, e.g. /dev/tty.usbmodem1101")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--burst", type=int, default=BURST,
                    help="frames captured per crossing (default %(default)s)")

    g = ap.add_argument_group("recognition")
    g.add_argument("--no-faces", action="store_true", help="camera only")
    g.add_argument("--gallery", default="data/faces.json")
    g.add_argument("--model", default="buffalo_sc",
                   help="buffalo_sc (15MB, fast) or buffalo_l (313MB, accurate)")
    g.add_argument("--det-size", type=int, default=640)
    g.add_argument("--threshold", type=float, default=0.40,
                   help="cosine similarity for a match (default %(default)s)")
    g.add_argument("--cooldown", type=float, default=30.0,
                   help="seconds before re-alerting the same person")
    g.add_argument("--threads", type=int, default=0)
    g.add_argument("--recognizer", default=None,
                   help="offload to a `faces.py serve` worker, e.g. http://host:8300")

    a = ap.add_argument_group("alerts")
    a.add_argument("--webhook", default=None, help="POST match JSON here")
    a.add_argument("--no-serial-alert", action="store_true",
                   help="recognise, but do not send <E/<K back to the MCU")

    args = ap.parse_args()

    OUT, CAM, BURST = args.out, args.camera, max(1, args.burst)
    WEBHOOK = args.webhook
    SERIAL_ALERTS = not args.no_serial_alert

    if not args.no_faces:
        setup_recognition(args)

    open_camera(CAM)

    if args.serial:
        threading.Thread(
            target=watch_serial, args=(args.serial, args.baud), daemon=True
        ).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\nready — POST http://localhost:{args.port}/snap")
    print(f"        UI   http://localhost:{args.port}/\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{_count} photos in {OUT}")
    finally:
        if _cap:
            _cap.release()


if __name__ == "__main__":
    main()

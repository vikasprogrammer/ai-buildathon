#!/usr/bin/env python3
"""Face enrolment and matching for TallyPoint.

The doorway already tells us *when* someone crossed. This tells us *who*.

That distinction is the whole reason this is feasible on the UNO Q: we run a
handful of inferences per crossing, not thirty per second. A quad Cortex-A53
cannot do continuous face video, but it can comfortably answer one question
every time the gates fire.

    python3 faces.py enroll --name Vikas photo.jpg [more.jpg ...]
    python3 faces.py list
    python3 faces.py remove --name Vikas
    python3 faces.py test frame.jpg          # who's in this picture?
    python3 faces.py bench frame.jpg         # how fast is this box?
    python3 faces.py serve                   # offload worker, see below

Two deployment shapes, same code:

  on-device   the UNO Q runs the model itself       (LocalRecognizer)
  offloaded   the UNO Q POSTs frames to a laptop    (RemoteRecognizer)

Start with on-device. If `bench` on the real board says it's too slow, run
`faces.py serve` on a laptop and point snap.py at it with --recognizer. Nothing
else changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np

# insightface leans on skimage, which is noisy about a deprecation we don't control.
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Tunables — grouped here so they can be adjusted at the venue without hunting.
# ---------------------------------------------------------------------------

GALLERY_PATH = "data/faces.json"
FACE_IMAGE_DIR = "data/faces"

#: Model pack. buffalo_sc = det_500m + w600k_mbf (MobileFaceNet), 15 MB total.
#: buffalo_l is ~4.6x slower and 313 MB — better accuracy, but it will not keep
#: up on the UNO Q. Measure with `faces.py bench` before reaching for it.
DEFAULT_MODEL = "buffalo_sc"

#: Detector input size. Bigger sees smaller/more distant faces; cost is roughly
#: quadratic. 640 is the safe default. Drop to 320 only if the camera is close
#: to the door — at 320 a face 3 m away is near the detector's floor.
DEFAULT_DET_SIZE = 640

#: Cosine similarity above which two embeddings are called the same person.
#: ArcFace embeddings are L2-normalised, so cosine == dot product.
#: Same person typically lands 0.40–0.80; different people −0.10–0.25.
#: 0.40 is deliberately a little strict — a false "that's Vikas" is worse than
#: a miss. Every score is logged so this can be tuned against real footage.
DEFAULT_THRESHOLD = 0.40

#: Detector confidence floor. Below this it's usually a face-like smudge.
DEFAULT_DET_THRESHOLD = 0.5

#: Ignore faces smaller than this (pixels, longest bbox edge). Tiny faces give
#: unreliable embeddings and are the main source of false matches.
MIN_FACE_PX = 50

#: Don't re-alert for the same person within this many seconds. Someone
#: standing in the doorway should not produce a stream of alerts.
DEFAULT_COOLDOWN_S = 30.0

EMBEDDING_DIM = 512


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Detected faces
# ---------------------------------------------------------------------------


@dataclass
class DetectedFace:
    """One face found in one frame."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    det_score: float
    embedding: np.ndarray  # L2-normalised, EMBEDDING_DIM floats
    crop: np.ndarray | None = None  # aligned 112x112 BGR, for the "enrol this?" flow

    @property
    def size_px(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(x2 - x1, y2 - y1)

    def to_json(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "det_score": round(float(self.det_score), 4),
            "embedding": [round(float(v), 6) for v in self.embedding],
        }

    @staticmethod
    def from_json(d: dict) -> "DetectedFace":
        return DetectedFace(
            bbox=tuple(d["bbox"]),  # type: ignore[arg-type]
            det_score=float(d["det_score"]),
            embedding=np.asarray(d["embedding"], dtype=np.float32),
        )


@dataclass
class Match:
    """The gallery's verdict on one detected face."""

    name: str | None  # None => no one over threshold
    score: float  # best cosine similarity seen
    runner_up: str | None = None
    runner_up_score: float = 0.0
    face: DetectedFace | None = None

    #: Best identity *regardless* of threshold. `name` is None below threshold,
    #: which would otherwise throw away the one fact you need to tune the
    #: threshold: who we nearly matched, and how close it got.
    top_name: str | None = None

    @property
    def matched(self) -> bool:
        return self.name is not None

    @property
    def margin(self) -> float:
        """Gap to the next-best identity. A thin margin means an ambiguous ID."""
        return self.score - self.runner_up_score

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "runner_up": self.runner_up,
            "runner_up_score": round(self.runner_up_score, 4),
            "margin": round(self.margin, 4),
            "bbox": list(self.face.bbox) if self.face else None,
        }


# ---------------------------------------------------------------------------
# Gallery — the enrolled watchlist
# ---------------------------------------------------------------------------


@dataclass
class Person:
    name: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    added: str = field(default_factory=_now_iso)
    alert: bool = True  # False => recognise but stay quiet
    note: str = ""


class Gallery:
    """Enrolled people, persisted as inspectable JSON.

    Embeddings are only comparable within one model pack, so the pack name is
    stored alongside them. Switching packs invalidates every enrolment — we
    refuse to match rather than silently return nonsense scores.
    """

    def __init__(self, path: str = GALLERY_PATH, model: str = DEFAULT_MODEL):
        self.path = path
        self.model = model
        self.people: dict[str, Person] = {}
        self._lock = threading.Lock()

    # -- persistence --------------------------------------------------------

    @classmethod
    def load(cls, path: str = GALLERY_PATH, model: str = DEFAULT_MODEL) -> "Gallery":
        g = cls(path, model)
        if not os.path.exists(path):
            return g
        with open(path) as fh:
            raw = json.load(fh)

        g.model = raw.get("model", model)
        for name, p in raw.get("people", {}).items():
            g.people[name] = Person(
                name=name,
                embeddings=[
                    np.asarray(e, dtype=np.float32) for e in p.get("embeddings", [])
                ],
                images=p.get("images", []),
                added=p.get("added", ""),
                alert=p.get("alert", True),
                note=p.get("note", ""),
            )
        return g

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "model": self.model,
            "dim": EMBEDDING_DIM,
            "saved": _now_iso(),
            "people": {
                p.name: {
                    "embeddings": [
                        [round(float(v), 6) for v in e] for e in p.embeddings
                    ],
                    "images": p.images,
                    "added": p.added,
                    "alert": p.alert,
                    "note": p.note,
                }
                for p in self.people.values()
            },
        }
        # Write via a temp file so a crash mid-save can't leave a truncated
        # gallery — re-enrolling everyone at a hackathon is a bad afternoon.
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)

    # -- mutation -----------------------------------------------------------

    def add(self, name: str, embedding: np.ndarray, image_path: str = "") -> Person:
        """Add one more reference embedding for `name`, creating them if new.

        Multiple references per person is the single cheapest accuracy win:
        different angles and lighting each cover a failure mode of the others.
        """
        with self._lock:
            person = self.people.get(name)
            if person is None:
                person = Person(name=name)
                self.people[name] = person
            person.embeddings.append(np.asarray(embedding, dtype=np.float32))
            if image_path:
                person.images.append(image_path)
            return person

    def remove(self, name: str) -> bool:
        with self._lock:
            return self.people.pop(name, None) is not None

    # -- matching -----------------------------------------------------------

    def match(
        self, embedding: np.ndarray, threshold: float = DEFAULT_THRESHOLD
    ) -> Match:
        """Best identity for this embedding, or Match(name=None).

        A person's score is the max over their reference embeddings — one good
        angle is enough, and averaging would let a bad reference drag down a
        genuine hit.
        """
        emb = np.asarray(embedding, dtype=np.float32)
        scored: list[tuple[float, str]] = []
        with self._lock:
            for person in self.people.values():
                if not person.embeddings:
                    continue
                best = max(float(np.dot(emb, ref)) for ref in person.embeddings)
                scored.append((best, person.name))

        if not scored:
            return Match(name=None, score=0.0)

        scored.sort(reverse=True)
        top_score, top_name = scored[0]
        runner_up, runner_up_score = (scored[1][1], scored[1][0]) if len(scored) > 1 else (None, 0.0)

        return Match(
            name=top_name if top_score >= threshold else None,
            top_name=top_name,
            score=top_score,
            runner_up=runner_up,
            runner_up_score=runner_up_score,
        )

    def __len__(self) -> int:
        return len(self.people)

    def __bool__(self) -> bool:
        # Without this, __len__ makes an empty gallery falsy, and every
        # `if gallery:` check silently means "if anyone is enrolled". A gallery
        # with no one in it is still a perfectly good gallery.
        return True

    @property
    def embedding_count(self) -> int:
        return sum(len(p.embeddings) for p in self.people.values())


# ---------------------------------------------------------------------------
# Alert gating
# ---------------------------------------------------------------------------


class AlertGate:
    """Per-person cooldown, so a loiterer doesn't machine-gun the alert channel.

    Deliberately separate from the gallery: whether we *recognise* someone and
    whether we *shout about it* are different questions, and only the second
    one should be rate-limited.
    """

    def __init__(self, cooldown_s: float = DEFAULT_COOLDOWN_S):
        self.cooldown_s = cooldown_s
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def should_alert(self, name: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            last = self._last.get(name)
            if last is not None and (now - last) < self.cooldown_s:
                return False
            self._last[name] = now
            return True

    def reset(self, name: str | None = None) -> None:
        with self._lock:
            if name is None:
                self._last.clear()
            else:
                self._last.pop(name, None)


# ---------------------------------------------------------------------------
# Recognisers
# ---------------------------------------------------------------------------


class LocalRecognizer:
    """Runs the ONNX models in this process.

    The model load costs ~2-15 s, so it happens once, lazily, on first use —
    importing this module stays cheap for the CLI paths that never infer.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        det_size: int = DEFAULT_DET_SIZE,
        det_threshold: float = DEFAULT_DET_THRESHOLD,
        threads: int = 0,
    ):
        self.model = model
        self.det_size = det_size
        self.det_threshold = det_threshold
        self.threads = threads
        self._app = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"local:{self.model}@{self.det_size}"

    def _ensure(self):
        if self._app is not None:
            return self._app
        with self._lock:
            if self._app is not None:
                return self._app
            try:
                from insightface.app import FaceAnalysis
            except ImportError as e:
                raise RuntimeError(
                    "insightface is not installed — run:\n"
                    "    pip install insightface onnxruntime"
                ) from e

            kwargs = {}
            if self.threads:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = self.threads
                so.inter_op_num_threads = 1
                kwargs["session_options"] = so

            t0 = time.time()
            # Only detection + recognition: the landmark/genderage models are
            # a pure download and memory tax for what we're doing.
            app = FaceAnalysis(
                name=self.model,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
                **kwargs,
            )
            app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size),
                        det_thresh=self.det_threshold)
            print(f"[faces] {self.model} ready in {time.time() - t0:.1f}s "
                  f"(det {self.det_size}px, thresh {self.det_threshold})",
                  file=sys.stderr)
            self._app = app
            return app

    def faces(self, image: np.ndarray, want_crops: bool = False) -> list[DetectedFace]:
        app = self._ensure()
        out: list[DetectedFace] = []
        for f in app.get(image):
            bbox = tuple(int(v) for v in f.bbox[:4])
            face = DetectedFace(
                bbox=bbox,  # type: ignore[arg-type]
                det_score=float(f.det_score),
                embedding=np.asarray(f.normed_embedding, dtype=np.float32),
            )
            if face.size_px < MIN_FACE_PX:
                continue
            if want_crops:
                face.crop = _crop_bbox(image, bbox)
            out.append(face)
        # Biggest face first: at a doorway, that's the person actually crossing
        # rather than someone in the background.
        out.sort(key=lambda f: f.size_px, reverse=True)
        return out

    def warm(self) -> None:
        """Load models and run one inference, so the first real crossing isn't slow."""
        self._ensure()
        self.faces(np.zeros((self.det_size, self.det_size, 3), dtype=np.uint8))


class RemoteRecognizer:
    """Offloads inference to a `faces.py serve` worker over HTTP.

    For when `bench` on the real board says on-device isn't fast enough. The
    UNO Q keeps owning the counting and the alerting; it just borrows someone
    else's CPU for the embedding.
    """

    def __init__(self, url: str, timeout: float = 8.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"remote:{self.url}"

    def faces(self, image: np.ndarray, want_crops: bool = False) -> list[DetectedFace]:
        import urllib.request

        import cv2

        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("could not encode frame for remote recogniser")

        req = urllib.request.Request(
            f"{self.url}/embed", data=buf.tobytes(),
            headers={"Content-Type": "image/jpeg"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())

        out = [DetectedFace.from_json(d) for d in payload.get("faces", [])]
        if want_crops:
            for f in out:
                f.crop = _crop_bbox(image, f.bbox)
        return out

    def warm(self) -> None:
        import urllib.request

        with urllib.request.urlopen(f"{self.url}/health", timeout=self.timeout) as r:
            r.read()


def _crop_bbox(image: np.ndarray, bbox: Sequence[int], pad: float = 0.25) -> np.ndarray:
    """Crop a face with some headroom — a tight bbox makes a poor thumbnail."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return image[y1:y2, x1:x2].copy()


def make_recognizer(
    remote: str | None = None,
    model: str = DEFAULT_MODEL,
    det_size: int = DEFAULT_DET_SIZE,
    det_threshold: float = DEFAULT_DET_THRESHOLD,
    threads: int = 0,
):
    if remote:
        return RemoteRecognizer(remote)
    return LocalRecognizer(model, det_size, det_threshold, threads)


# ---------------------------------------------------------------------------
# Identify: the operation snap.py actually calls
# ---------------------------------------------------------------------------


def identify(
    recognizer,
    frames: Iterable[np.ndarray],
    gallery: Gallery,
    threshold: float = DEFAULT_THRESHOLD,
    want_crops: bool = True,
) -> list[Match]:
    """Best identification across a burst of frames.

    A single frame at the moment of crossing is a coin flip — motion blur, a
    turned head, a blink. Several frames cost little (the models are small and
    we only do this on a gate event) and turn most coin flips into hits.

    Faces are collapsed by identity across frames, keeping the highest-scoring
    sighting of each, so one person walking through yields one result, not one
    per frame.
    """
    best: dict[str, Match] = {}
    unknowns: list[Match] = []

    for frame in frames:
        for face in recognizer.faces(frame, want_crops=want_crops):
            m = gallery.match(face.embedding, threshold)
            m.face = face
            if m.matched:
                assert m.name is not None
                prior = best.get(m.name)
                if prior is None or m.score > prior.score:
                    best[m.name] = m
            else:
                unknowns.append(m)

    results = sorted(best.values(), key=lambda m: m.score, reverse=True)

    # Report at most one unknown — the biggest, best-detected face we couldn't
    # place. Enough to offer "enrol this person?" without spamming the feed.
    if unknowns:
        unknowns.sort(
            key=lambda m: (m.face.size_px if m.face else 0,
                           m.face.det_score if m.face else 0),
            reverse=True,
        )
        results.append(unknowns[0])

    return results


# ---------------------------------------------------------------------------
# Enrolment
# ---------------------------------------------------------------------------


class EnrolmentError(Exception):
    pass


def enroll_image(
    recognizer,
    gallery: Gallery,
    name: str,
    image: np.ndarray,
    save_as: str | None = None,
) -> tuple[Person, DetectedFace]:
    """Enrol the most prominent face in `image` as `name`.

    Refuses ambiguous input rather than guessing: no face and we have nothing;
    a very small face gives a weak embedding that will haunt every later match.
    Multiple faces is fine — we take the biggest, which for a portrait is the
    subject.
    """
    faces = recognizer.faces(image, want_crops=True)
    if not faces:
        raise EnrolmentError(
            "no face found — use a clear, front-on, well-lit photo"
        )

    face = faces[0]  # recognizer.faces() sorts biggest-first
    if face.size_px < MIN_FACE_PX * 2:
        raise EnrolmentError(
            f"face is only {face.size_px}px across; use a closer or higher-"
            f"resolution photo (want at least {MIN_FACE_PX * 2}px)"
        )

    path = ""
    if save_as:
        import cv2

        os.makedirs(os.path.dirname(save_as) or ".", exist_ok=True)
        cv2.imwrite(save_as, face.crop if face.crop is not None else image)
        path = save_as

    person = gallery.add(name, face.embedding, path)
    return person, face


def enrolment_path(name: str, index: int) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return os.path.join(FACE_IMAGE_DIR, f"{safe or 'person'}-{index:03d}.jpg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_image(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"could not read image: {path}")
    return img


def cmd_enroll(args) -> int:
    import cv2  # noqa: F401  (fail early if OpenCV is missing)

    gallery = Gallery.load(args.gallery, args.model)
    _check_model_match(gallery, args.model)
    rec = make_recognizer(args.recognizer, args.model, args.det_size)

    added = 0
    for src in args.images:
        img = _read_image(src)
        idx = len(gallery.people.get(args.name, Person(args.name)).images) + added
        try:
            person, face = enroll_image(
                rec, gallery, args.name, img, enrolment_path(args.name, idx)
            )
        except EnrolmentError as e:
            print(f"  ✗ {src}: {e}")
            continue
        added += 1
        print(f"  ✓ {src}: face {face.size_px}px, det {face.det_score:.2f} "
              f"→ {person.name} (now {len(person.embeddings)} reference(s))")

    if not added:
        print("nothing enrolled")
        return 1

    gallery.model = args.model
    gallery.save()
    print(f"\n{args.name} enrolled — {len(gallery)} person(s), "
          f"{gallery.embedding_count} reference(s) in {gallery.path}")

    if added == 1 and len(gallery.people[args.name].embeddings) == 1:
        print("tip: enrol 2-3 more photos at different angles/lighting; it is "
              "the cheapest accuracy win available.")
    return 0


def cmd_list(args) -> int:
    gallery = Gallery.load(args.gallery, args.model)
    if not gallery.people:
        print(f"no one enrolled ({gallery.path} is empty or missing)")
        return 0
    print(f"gallery: {gallery.path}   model: {gallery.model}\n")
    print(f"{'name':<24} {'refs':>4}  {'alert':<6} added")
    print("-" * 64)
    for p in gallery.people.values():
        print(f"{p.name:<24} {len(p.embeddings):>4}  "
              f"{'yes' if p.alert else 'no':<6} {p.added}")
    return 0


def cmd_remove(args) -> int:
    gallery = Gallery.load(args.gallery, args.model)
    if not gallery.remove(args.name):
        print(f"{args.name} is not enrolled")
        return 1
    gallery.save()
    print(f"removed {args.name} — {len(gallery)} person(s) left")
    return 0


def cmd_test(args) -> int:
    gallery = Gallery.load(args.gallery, args.model)
    _check_model_match(gallery, args.model)
    rec = make_recognizer(args.recognizer, args.model, args.det_size)
    img = _read_image(args.image)

    t0 = time.perf_counter()
    results = identify(rec, [img], gallery, args.threshold, want_crops=False)
    dt = (time.perf_counter() - t0) * 1000

    print(f"\n{args.image}: {img.shape[1]}x{img.shape[0]} in {dt:.0f}ms")
    if not results:
        print("  no faces detected")
        return 0
    for m in results:
        who = m.name or "UNKNOWN"
        bbox = m.face.bbox if m.face else ()
        print(f"  {who:<20} score {m.score:.3f}  margin {m.margin:+.3f}  bbox {bbox}")
        if not m.matched and gallery.people:
            print(f"    (closest was {m.top_name or '—'} at {m.score:.3f}, "
                  f"threshold {args.threshold})")
    return 0


def cmd_bench(args) -> int:
    """Measure this machine. Run this on the UNO Q before trusting any estimate."""
    img = _read_image(args.image)
    print(f"image {img.shape[1]}x{img.shape[0]}   reps {args.reps}")
    print(f"{'model':<12} {'det':>5} {'thr':>4} {'faces':>6} {'median':>9} {'p90':>8}")
    print("-" * 50)

    for model in args.models:
        for det in args.det_sizes:
            rec = LocalRecognizer(model, det, threads=args.threads)
            try:
                faces = rec.faces(img)  # warm up
            except RuntimeError as e:
                print(f"{model:<12} {det:>5}  FAILED: {e}")
                continue
            times = []
            for _ in range(args.reps):
                t = time.perf_counter()
                rec.faces(img)
                times.append((time.perf_counter() - t) * 1000)
            print(f"{model:<12} {det:>5} {args.threads or 'all':>4} {len(faces):>6} "
                  f"{np.median(times):>8.0f}ms {np.percentile(times, 90):>7.0f}ms")

    print("\nBudget: one crossing costs roughly (burst size x median). With the "
          "default burst of 3, keep median under ~200ms to stay comfortably "
          "inside a person's walk through the door.")
    return 0


def cmd_serve(args) -> int:
    """Offload worker: POST a JPEG to /embed, get faces + embeddings back."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import cv2

    rec = LocalRecognizer(args.model, args.det_size, threads=args.threads)
    print("warming models...")
    rec.warm()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/") not in ("", "/health"):
                self.send_error(404)
                return
            self._json({"ok": True, "recognizer": rec.name})

        def do_POST(self):
            if self.path.rstrip("/") != "/embed":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            if not n:
                self.send_error(400, "empty body")
                return
            raw = self.rfile.read(n)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self.send_error(400, "could not decode image")
                return
            t0 = time.perf_counter()
            faces = rec.faces(img)
            dt = (time.perf_counter() - t0) * 1000
            print(f"  {len(faces)} face(s) in {dt:.0f}ms")
            self._json({"faces": [f.to_json() for f in faces], "ms": round(dt, 1)})

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"recogniser ready — POST http://0.0.0.0:{args.port}/embed")
    print(f"point snap.py at it with:  --recognizer http://<this-host>:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


def _check_model_match(gallery: Gallery, model: str) -> None:
    """Embeddings from different packs are not comparable — say so loudly."""
    if gallery.people and gallery.model != model:
        print(
            f"\n!! gallery was built with '{gallery.model}' but you are using "
            f"'{model}'.\n"
            f"   Embeddings from different model packs are not comparable and "
            f"the scores\n   would be meaningless. Either use --model "
            f"{gallery.model}, or re-enrol everyone.\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gallery", default=GALLERY_PATH)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"insightface pack (default {DEFAULT_MODEL})")
    ap.add_argument("--det-size", type=int, default=DEFAULT_DET_SIZE)
    ap.add_argument("--recognizer", default=None,
                    help="offload to a `faces.py serve` worker, e.g. http://host:8300")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("enroll", help="add a person from one or more photos")
    p.add_argument("--name", required=True)
    p.add_argument("images", nargs="+")
    p.set_defaults(fn=cmd_enroll)

    p = sub.add_parser("list", help="show who is enrolled")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("remove", help="drop a person")
    p.add_argument("--name", required=True)
    p.set_defaults(fn=cmd_remove)

    p = sub.add_parser("test", help="identify faces in a still image")
    p.add_argument("image")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.set_defaults(fn=cmd_test)

    p = sub.add_parser("bench", help="measure inference speed on this machine")
    p.add_argument("image")
    p.add_argument("--models", nargs="+", default=["buffalo_sc", "buffalo_l"])
    p.add_argument("--det-sizes", nargs="+", type=int, default=[640, 320])
    p.add_argument("--threads", type=int, default=0, help="0 = let onnxruntime decide")
    p.add_argument("--reps", type=int, default=8)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("serve", help="run as an offload worker for another host")
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--threads", type=int, default=0)
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

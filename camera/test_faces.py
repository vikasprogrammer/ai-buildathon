#!/usr/bin/env python3
"""Tests for the face pipeline.

    python3 test_faces.py

Runs without a camera and without touching a live server: the capture device is
stubbed with real JPEGs from data/snaps/, so the whole path — crossing, burst,
recognition, match, event — is exercised for real, just deterministically.

The model itself is only loaded by the integration tests. If insightface is
missing those skip and the pure-logic tests still run.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faces  # noqa: E402

SNAP_DIR = "data/snaps"


def unit(*parts) -> np.ndarray:
    """A normalised embedding, so cosine similarity behaves like the real thing."""
    v = np.zeros(faces.EMBEDDING_DIM, dtype=np.float32)
    for i, p in enumerate(parts):
        v[i] = p
    n = np.linalg.norm(v)
    return v / n if n else v


def sample_snaps(n: int = 4) -> list[str]:
    if not os.path.isdir(SNAP_DIR):
        return []
    got = sorted(f for f in os.listdir(SNAP_DIR) if f.lower().endswith(".jpg"))
    return [os.path.join(SNAP_DIR, f) for f in got[:n]]


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------


class TestGallery(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "faces.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_empty_gallery_matches_nothing(self):
        g = faces.Gallery(self.path)
        m = g.match(unit(1))
        self.assertIsNone(m.name)
        self.assertFalse(m.matched)

    def test_exact_match_scores_one(self):
        g = faces.Gallery(self.path)
        e = unit(1, 2, 3)
        g.add("Ada", e)
        m = g.match(e)
        self.assertEqual(m.name, "Ada")
        self.assertAlmostEqual(m.score, 1.0, places=5)

    def test_orthogonal_embedding_is_not_a_match(self):
        g = faces.Gallery(self.path)
        g.add("Ada", unit(1, 0))
        m = g.match(unit(0, 1))
        self.assertIsNone(m.name)
        self.assertAlmostEqual(m.score, 0.0, places=5)

    def test_threshold_is_the_boundary(self):
        g = faces.Gallery(self.path)
        g.add("Ada", unit(1, 0))
        probe = unit(1, 1)  # cos = 0.7071
        self.assertEqual(g.match(probe, threshold=0.70).name, "Ada")
        self.assertIsNone(g.match(probe, threshold=0.71).name)

    def test_best_of_several_references_wins(self):
        """One good angle should be enough — averaging would let a weak
        reference drag down a genuine hit."""
        g = faces.Gallery(self.path)
        g.add("Ada", unit(1, 0))
        g.add("Ada", unit(0, 1))
        m = g.match(unit(0, 1))
        self.assertEqual(m.name, "Ada")
        self.assertAlmostEqual(m.score, 1.0, places=5)

    def test_runner_up_and_margin_reported(self):
        g = faces.Gallery(self.path)
        g.add("Ada", unit(1, 0))
        g.add("Bob", unit(1, 1))
        m = g.match(unit(1, 0), threshold=0.5)
        self.assertEqual(m.name, "Ada")
        self.assertEqual(m.runner_up, "Bob")
        self.assertGreater(m.margin, 0)
        self.assertAlmostEqual(m.score - m.runner_up_score, m.margin, places=5)

    def test_roundtrip_preserves_scores(self):
        g = faces.Gallery(self.path, model="buffalo_sc")
        e = unit(1, 2, 3)
        g.add("Ada", e, "ref.jpg")
        g.save()

        g2 = faces.Gallery.load(self.path)
        self.assertEqual(g2.model, "buffalo_sc")
        self.assertEqual(len(g2), 1)
        self.assertEqual(g2.people["Ada"].images, ["ref.jpg"])
        self.assertAlmostEqual(g2.match(e).score, 1.0, places=4)

    def test_save_is_atomic(self):
        """A crash mid-save must not leave a truncated gallery."""
        g = faces.Gallery(self.path)
        g.add("Ada", unit(1))
        g.save()
        g.add("Bob", unit(0, 1))
        g.save()
        with open(self.path) as fh:
            json.load(fh)  # parses => not truncated
        self.assertFalse(os.path.exists(self.path + ".tmp"))
        self.assertEqual(len(faces.Gallery.load(self.path)), 2)

    def test_remove(self):
        g = faces.Gallery(self.path)
        g.add("Ada", unit(1))
        self.assertTrue(g.remove("Ada"))
        self.assertFalse(g.remove("Ada"))
        self.assertIsNone(g.match(unit(1)).name)

    def test_missing_file_loads_empty(self):
        g = faces.Gallery.load(os.path.join(self.dir, "nope.json"))
        self.assertEqual(len(g), 0)


# ---------------------------------------------------------------------------
# Alert gating
# ---------------------------------------------------------------------------


class TestAlertGate(unittest.TestCase):
    def test_first_sighting_alerts(self):
        self.assertTrue(faces.AlertGate(30).should_alert("Ada", now=0.0))

    def test_repeat_inside_cooldown_is_suppressed(self):
        """Someone standing in the doorway must not machine-gun the alert."""
        gate = faces.AlertGate(30)
        self.assertTrue(gate.should_alert("Ada", now=0.0))
        self.assertFalse(gate.should_alert("Ada", now=5.0))
        self.assertFalse(gate.should_alert("Ada", now=29.9))

    def test_cooldown_expires(self):
        gate = faces.AlertGate(30)
        gate.should_alert("Ada", now=0.0)
        self.assertTrue(gate.should_alert("Ada", now=30.1))

    def test_cooldown_is_per_person(self):
        gate = faces.AlertGate(30)
        gate.should_alert("Ada", now=0.0)
        self.assertTrue(gate.should_alert("Bob", now=1.0))

    def test_reset_rearms(self):
        gate = faces.AlertGate(30)
        gate.should_alert("Ada", now=0.0)
        gate.reset("Ada")
        self.assertTrue(gate.should_alert("Ada", now=1.0))


# ---------------------------------------------------------------------------
# identify() across a burst
# ---------------------------------------------------------------------------


class FakeRecognizer:
    """Returns scripted faces per frame, so burst behaviour is testable."""

    name = "fake"

    def __init__(self, per_frame):
        self.per_frame = per_frame
        self.calls = 0

    def faces(self, frame, want_crops=False):
        out = self.per_frame[self.calls % len(self.per_frame)]
        self.calls += 1
        return out

    def warm(self):
        pass


def face(emb, score=0.9, size=200):
    return faces.DetectedFace(bbox=(0, 0, size, size), det_score=score, embedding=emb)


class TestIdentify(unittest.TestCase):
    def setUp(self):
        self.g = faces.Gallery(tempfile.mktemp())
        self.ada = unit(1, 0)
        self.g.add("Ada", self.ada)

    def test_one_person_across_frames_yields_one_result(self):
        """A burst must not report the same person three times."""
        rec = FakeRecognizer([[face(self.ada)], [face(self.ada)], [face(self.ada)]])
        out = faces.identify(rec, [None] * 3, self.g, want_crops=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].name, "Ada")

    def test_a_face_missed_in_some_frames_is_still_found(self):
        """The whole point of the burst: frame 1 misses, frame 2 saves it."""
        rec = FakeRecognizer([[], [face(self.ada)], []])
        out = faces.identify(rec, [None] * 3, self.g, want_crops=False)
        self.assertEqual([m.name for m in out], ["Ada"])

    def test_no_faces_at_all_yields_nothing(self):
        rec = FakeRecognizer([[], [], []])
        self.assertEqual(faces.identify(rec, [None] * 3, self.g, want_crops=False), [])

    def test_unknown_is_reported_once(self):
        stranger = unit(0, 1)
        rec = FakeRecognizer([[face(stranger)], [face(stranger)]])
        out = faces.identify(rec, [None] * 2, self.g, want_crops=False)
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].matched)

    def test_known_and_unknown_together(self):
        rec = FakeRecognizer([[face(self.ada), face(unit(0, 1))]])
        out = faces.identify(rec, [None], self.g, want_crops=False)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].name, "Ada")     # matches sort first
        self.assertFalse(out[-1].matched)

    def test_best_scoring_sighting_is_kept(self):
        near = unit(1, 0.05)
        rec = FakeRecognizer([[face(near)], [face(self.ada)]])
        out = faces.identify(rec, [None] * 2, self.g, want_crops=False)
        self.assertAlmostEqual(out[0].score, 1.0, places=4)


# ---------------------------------------------------------------------------
# HTTP surface — real server, stubbed camera
# ---------------------------------------------------------------------------


class FakeCap:
    """Stands in for cv2.VideoCapture, replaying real JPEGs off disk."""

    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def isOpened(self):
        return True

    def read(self):
        if not self.frames:
            return False, None
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        return True, f.copy()

    def set(self, *a):
        return True

    def release(self):
        pass


def has_model() -> bool:
    try:
        import insightface  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(sample_snaps(), f"no sample JPEGs in {SNAP_DIR}")
@unittest.skipUnless(has_model(), "insightface not installed")
class TestServer(unittest.TestCase):
    """Drives the real handler end to end: enrol, cross, match, event."""

    @classmethod
    def setUpClass(cls):
        import cv2

        import snap

        cls.snap = snap
        cls.dir = tempfile.mkdtemp()
        cls.paths = sample_snaps(4)
        frames = [cv2.imread(p) for p in cls.paths]
        frames = [f for f in frames if f is not None]
        assert frames, "could not read any sample JPEGs"

        snap.OUT = os.path.join(cls.dir, "snaps")
        snap.UNKNOWN_DIR = os.path.join(cls.dir, "unknown")
        snap.BURST = 2
        snap._cap = FakeCap(frames)
        snap._count = 0

        faces.FACE_IMAGE_DIR = os.path.join(cls.dir, "refs")

        class Args:
            gallery = os.path.join(cls.dir, "faces.json")
            model = "buffalo_sc"
            det_size = 640
            threshold = 0.40
            cooldown = 30.0
            threads = 0
            recognizer = None

        assert snap.setup_recognition(Args()), "recognition failed to start"

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), snap.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        shutil.rmtree(cls.dir, ignore_errors=True)

    # -- helpers ------------------------------------------------------------

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path):
        with urllib.request.urlopen(self.url(path), timeout=20) as r:
            return r.status, r.read()

    def get_json(self, path):
        return json.loads(self.get(path)[1])

    def post(self, path, body=b"", ctype="application/json"):
        req = urllib.request.Request(
            self.url(path), data=body, headers={"Content-Type": ctype}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")

    def wait_for(self, kind, after=0, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            for e in self.snap.BUS.history():
                if e["kind"] == kind and e["id"] > after:
                    return e
            time.sleep(0.05)
        kinds = [e["kind"] for e in self.snap.BUS.history()]
        self.fail(f"no {kind!r} event within {timeout}s (saw {kinds})")

    # -- tests --------------------------------------------------------------

    def test_01_pages_render(self):
        for path, marker in [("/", b"Enrol a face"), ("/wall", b"WATCHING THE DOOR")]:
            status, body = self.get(path)
            self.assertEqual(status, 200, path)
            self.assertIn(marker, body, path)

    def test_02_status_reports_recognition_on(self):
        s = self.get_json("/status")
        self.assertTrue(s["recognition"])
        self.assertEqual(s["burst"], 2)
        self.assertIn("buffalo_sc", s["recognizer"])

    def test_03_enroll_from_upload(self):
        with open(self.paths[0], "rb") as fh:
            status, body = self.post("/enroll?name=Ada", fh.read(), "image/jpeg")
        self.assertEqual(status, 200)
        self.assertEqual(body["name"], "Ada")
        self.assertEqual(body["references"], 1)
        self.assertGreater(body["face_px"], faces.MIN_FACE_PX)

        names = [p["name"] for p in self.get_json("/people")["people"]]
        self.assertIn("Ada", names)

    def test_04_enroll_rejects_a_faceless_image(self):
        import cv2

        blank = os.path.join(self.dir, "blank.jpg")
        cv2.imwrite(blank, np.zeros((480, 640, 3), dtype=np.uint8))
        with open(blank, "rb") as fh:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.post("/enroll?name=Ghost", fh.read(), "image/jpeg")
        self.assertEqual(cm.exception.code, 422)

    def test_05_enroll_needs_a_name(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("/enroll", b"x", "image/jpeg")
        self.assertEqual(cm.exception.code, 400)

    def test_06_crossing_identifies_the_enrolled_person(self):
        """The whole point: cross the door, get told who it was."""
        before = self.snap.BUS._seq
        status, body = self.post("/snap")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

        e = self.wait_for("match", after=before)
        self.assertEqual(e["name"], "Ada")
        self.assertGreaterEqual(e["score"], 0.40)
        self.assertTrue(e["alerted"], "first sighting should alert")
        self.assertTrue(os.path.exists(os.path.join(self.snap.OUT, e["snap"])))

    def test_07_second_crossing_is_recognised_but_not_re_alerted(self):
        before = self.snap.BUS._seq
        self.post("/snap")
        e = self.wait_for("match", after=before)
        self.assertEqual(e["name"], "Ada")
        self.assertFalse(e["alerted"], "cooldown should suppress the repeat")

    def test_08_snap_returns_before_recognition_finishes(self):
        """Recognition must stay off the trigger path."""
        t0 = time.perf_counter()
        self.post("/snap")
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed, 900, f"/snap blocked for {elapsed:.0f}ms")

    def test_09_events_stream_replays_history(self):
        req = urllib.request.Request(self.url("/events"))
        with urllib.request.urlopen(req, timeout=10) as r:
            self.assertEqual(r.status, 200)
            self.assertIn("text/event-stream", r.headers["Content-Type"])
            line = r.readline().decode()
            self.assertTrue(line.startswith("data: "), line)
            json.loads(line[6:])

    def test_10_image_routes_serve_snaps(self):
        name = os.listdir(self.snap.OUT)[0]
        status, body = self.get(f"/img/snap/{name}")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"\xff\xd8"), "not a JPEG")

    def test_11_image_routes_reject_traversal(self):
        for probe in ["..%2f..%2fsnap.py", "../../snap.py", "....//snap.py"]:
            with self.assertRaises(urllib.error.HTTPError, msg=probe) as cm:
                self.get(f"/img/snap/{probe}")
            self.assertEqual(cm.exception.code, 404, probe)

    def test_12_delete_removes_from_the_watchlist(self):
        req = urllib.request.Request(self.url("/people?name=Ada"), method="DELETE")
        with urllib.request.urlopen(req, timeout=10) as r:
            self.assertEqual(r.status, 200)
        self.assertEqual(self.get_json("/people")["people"], [])

    def test_13_unknown_face_after_removal(self):
        """With the gallery empty, the same person now reads as a stranger."""
        before = self.snap.BUS._seq
        self.post("/snap")
        e = self.wait_for("unknown", after=before)
        self.assertIsNone(e["closest"])

    def test_14_unknown_routes_404(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/nope")
        self.assertEqual(cm.exception.code, 404)


class TestSafeJoin(unittest.TestCase):
    def test_traversal_is_refused(self):
        import snap

        self.assertIsNone(snap._safe_join("data/snaps", "../../snap.py"))
        self.assertIsNone(snap._safe_join("data/snaps", "/etc/passwd"))
        self.assertIsNone(snap._safe_join("", "anything.jpg"))
        self.assertIsNone(snap._safe_join("data/snaps", "does-not-exist.jpg"))

    def test_real_file_resolves(self):
        import snap

        got = sample_snaps(1)
        if not got:
            self.skipTest("no sample snaps")
        name = os.path.basename(got[0])
        self.assertIsNotNone(snap._safe_join(SNAP_DIR, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)

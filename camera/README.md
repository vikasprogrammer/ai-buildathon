# Camera — who walked in

Takes a photo each time the counter reports someone entering, then identifies
them against an enrolled gallery and alerts.

The camera is not wired to the Arduino. It plugs into a computer running this
script, and the sketch's existing serial output is the trigger.

## How it hooks in

`visitor_counter.ino` already prints this from `report()` the moment the IR state
machine decides direction:

```
ENTRY  |  Occupancy: 3  |  Total visitors: 7
```

This watches for a line starting with `ENTRY` and captures a short burst of
frames. `EXIT` and everything else is printed and ignored.

```
IR gates -> sketch decides ENTRY -> Serial -> snap.py -> photo -> faces.py -> "that's Vikas" -> <K beep
```

Recognition runs **off the trigger path**: the crossing captures and returns
immediately, and a worker thread identifies. The doorway never waits on a model.

## Running it

```bash
pip install opencv-python pyserial insightface onnxruntime

ls /dev/tty.usbmodem*
python3 snap.py --serial /dev/tty.usbmodem1101 --baud 9600
```

> **Baud:** the sketch is currently `Serial.begin(9600)` while everything else in
> this project uses 115200. Until `visitor_counter.ino` is changed, pass
> `--baud 9600`. After it changes, drop the flag.

Then open <http://localhost:8200/> to enrol faces and watch the live feed, or
<http://localhost:8200/wall> for a full-screen display.

## Enrolling

From the UI, or from the command line:

```bash
python3 faces.py enroll --name Vikas photo1.jpg photo2.jpg
python3 faces.py list
python3 faces.py remove --name Vikas
python3 faces.py test frame.jpg      # who is in this picture?
```

Two or three photos per person at different angles is the cheapest accuracy win
available. Anyone unrecognised is saved as a crop so they can be enrolled after
the fact — the natural way to build a watchlist is to point at someone who
already walked through.

## Talking back to the board

On a confirmed match, this sends the MCU:

```
<E VIKAS     name, up to 8 chars — for the log and a status blink
<K           acknowledge, sound a short tone
```

The sketch does not read serial input yet, so these currently go nowhere — see
`docs/FIRMWARE_PROMPT.md`. Nothing breaks meanwhile; a tolerant parser ignores
them. Suppress with `--no-serial-alert`.

## Manual trigger / fallback

```bash
curl -X POST localhost:8200/snap    # fire the whole pipeline now
curl localhost:8200/status          # counts, recogniser, queue depth
```

Worth having in a terminal during a demo. If the IR gates misbehave, the camera
half still demos on its own. `--serial` is optional; without it this is
HTTP-only.

## Speed

`buffalo_sc` is the default: 15 MB, ~74 ms warm for a 3-frame burst on a laptop.
**Measure on the real board before trusting that** — the UNO Q is much slower:

```bash
python3 faces.py bench data/snaps/some.jpg
```

If the median is over ~200 ms, run `python3 faces.py serve` on a laptop and point
the board at it with `--recognizer http://laptop:8300`. Nothing else changes.

If crossings outrun the model, jobs are dropped and it says so — a silent backlog
would report the wrong person for the wrong crossing, which is worse.

## Notes

- The camera is held open rather than opened per shot; opening costs ~1 s, long
  enough to miss the person.
- One photo per crossing is a coin flip — motion blur, a turned head, a blink.
  Three frames cost ~66 ms of camera time and turn most coin flips into hits.
- Alerts are rate-limited per person (`--cooldown`, default 30 s) so a loiterer
  doesn't machine-gun the channel.
- `data/` is gitignored. It holds photos of real people and their face
  embeddings, and this repo is public.
- Don't run two instances against one gallery — saves rewrite the whole file from
  memory, so the second process silently clobbers the first one's enrolments.

## Tests

```bash
python3 -m pytest test_faces.py -q      # 37 tests, real models, no hardware
```

They stub the camera with sample JPEGs and redirect all output into temp dirs, so
they neither need the board nor touch `data/`.

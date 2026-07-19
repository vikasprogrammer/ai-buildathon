# Camera — snapshot on entry

Takes a photo each time the counter reports someone walking **in**.

The camera is not wired to the Arduino. It plugs into a computer running this
script, and the sketch's existing serial output is the trigger — no firmware
changes were needed.

## How it hooks in

`visitor_counter.ino` already prints this from `report()` the moment the IR
state machine decides direction:

```
ENTRY  |  Occupancy: 3  |  Total visitors: 7
```

This script watches the serial port for a line starting with `ENTRY` and grabs
the current camera frame. `EXIT` and everything else is printed and ignored.

```
Two IR gates -> sketch decides ENTRY -> Serial.println -> snap.py -> JPEG
```

## Running it

```bash
pip install pyserial opencv-python

ls /dev/tty.usbmodem*                              # find the board
python3 snap.py --serial /dev/tty.usbmodem1101     # macOS / Linux laptop
```

Photos land in `data/snaps/`, named `YYYYMMDD-HHMMSS-mmm_in.jpg` — timestamp
sortable.

Options: `--camera 0` (which webcam), `--port 8200` (HTTP port), `--out DIR`,
`--baud 115200`.

## Manual trigger / fallback

It also listens on HTTP, so a photo can be forced without the gates:

```bash
curl -X POST localhost:8200/snap    # take a photo now
curl localhost:8200                 # how many so far
```

Worth having in a terminal during a demo — if the IR gates misbehave, the camera
half still demos on its own.

`--serial` is optional. Without it the script is HTTP-only.

## Notes

- It holds the camera open rather than opening it per shot. Opening costs ~1 s,
  which is long enough to miss the person entirely.
- The first few frames off a webcam are dark or green, so it discards five at
  startup.
- The serial reader reconnects on its own. Unplugging the board does not kill
  the script.
- Latency matters: the frame is grabbed when the `ENTRY` line arrives, so the
  sketch should keep printing it immediately on detection, before any blocking
  work. It currently does.
- Tested end-to-end against a virtual serial port feeding real sketch output,
  and against a Logitech C270. Not yet tested against the physical board.

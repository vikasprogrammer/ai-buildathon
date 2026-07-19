# Prompt for the counting-firmware team

Paste everything below the line into Claude Code in the `ai-buildathon` repo.

> Written after reading `visitor_counter.ino` at commit `c03d980`. This is **not**
> a rebuild — the sketch works. It is four specific changes, two of which are
> one-liners, plus two counting bugs worth fixing if time allows.

---

## Context

I have a working visitor counter on the **Arduino UNO Q** (`visitor_counter.ino`).
Two IR sensors on D2/D3 detect direction, occupancy shows on a 7-segment display,
a red LED and buzzer fire at capacity, and an ESP32-C6 (Glyph) sends encoder and
RFID events over WiFi to a Python brick that calls my RPCs via Arduino Bridge.

**All of that works. Do not rewrite it.**

A teammate has added a camera. It runs on a laptop over USB, watches my serial
output, and takes a photo of whoever walks in — then runs face recognition and
tells me who it was. My sketch needs four changes to work with it.

## The integration, in one line

The camera triggers on the line my `report()` already prints:

```
ENTRY  |  Occupancy: 3  |  Total visitors: 7
```

It matches on the prefix `ENTRY`. **Do not change that prefix or move it later in
the line.** `EXIT` and every other line is read and ignored. No new event type is
needed — my existing output is the contract.

---

## Change 1 — baud rate (one line, do this first)

```cpp
Serial.begin(9600);     // current
Serial.begin(115200);   // needs to be this
```

The camera side reads at 115200. My ESP32 sketch is already at 115200, so this
also makes the two boards consistent. **Nothing works until this matches** — at
the wrong baud the camera sees garbage bytes or nothing at all, which looks like
a broken camera rather than a config mismatch.

## Change 2 — read the serial input (currently there is none)

My sketch calls `Serial.println()` but never `Serial.read()`. The camera sends a
line back when it recognises a face, and right now it goes nowhere.

Two commands, `\n`-terminated, `<` sentinel:

| Line | Meaning |
|---|---|
| `<E VIKAS` | A recognised person's name, up to 8 chars, A–Z and 0–9 |
| `<K` | Acknowledge — sound a short confirmation tone |

What I want on receipt:
- `<K` → short confirmation beep, distinct from the capacity-breach alert.
- `<E <name>` → I have one 7-segment digit, so I cannot render a name. Log it to
  Serial, and flash the blue/status LED briefly. The **beep is the part a human
  notices**; the name is for the log.

Requirements:
- **Non-blocking.** Accumulate bytes into a small buffer across `loop()`
  iterations, act on `\n`. Never `while (!Serial.available())`.
- **Tolerant.** Ignore unknown commands, partial lines, and garbage rather than
  faulting. More command types may come later.
- Cap the buffer (say 32 bytes) and discard overlong lines so a noisy link can't
  overflow it.

## Change 3 — a fast walker is silently missed (real counting bug)

In `handleSensors()`, the `IDLE` case only advances on:

```cpp
if (ir1 && !ir2)       { state = WAIT_IR2_ENTRY; ... }
else if (ir2 && !ir1)  { state = WAIT_IR1_EXIT;  ... }
```

If someone breaks **both** beams within one `loop()` iteration — likely, since
the sensors are close together and `loop()` also polls the encoder, services
Bridge, and refreshes the display — neither branch fires. The passage is dropped
with no count, no display change, and **no photo**.

Fix: latch whichever sensor is seen first on a *transition*, rather than
requiring the other to still be clear. Track the previous reading of each pin and
act on the rising edge of "beam broken", so a both-broken sample still records
which one changed first.

This is the highest-value fix here — it's a silent miss, so nobody notices it
during testing, and it undercounts exactly when the doorway is busy.

## Change 4 — `waitUntilClear()` freezes everything for up to 2 s

```cpp
void waitUntilClear() {
  unsigned long t = millis();
  while (millis() - t < 2000) { ... }   // busy-wait
}
```

While this spins, the display doesn't update, the encoder and button don't
respond, Bridge RPCs aren't serviced, and incoming serial isn't read. Anyone
loitering in the doorway freezes the whole device for two seconds.

Fix: make it a state in the existing machine — add `WAIT_CLEAR`, enter it after a
count, and leave it when both beams read clear or a timeout expires. Same
behaviour, no blocking.

Note the ordering that already exists and must stay: `report()` is called
**before** `waitUntilClear()`, so the `ENTRY` line goes out immediately. The
camera grabs the current frame the moment it sees that line — a late line means a
photo of an empty doorway. Keep printing before any waiting.

---

## Priority

1. **Change 1** (baud) — one line, nothing works without it.
2. **Change 3** (missed passages) — silent undercount, and a missed photo.
3. **Change 2** (serial input) — enables the recognition beep.
4. **Change 4** (non-blocking clear) — quality; visible if anyone loiters.

If time runs out, 1 and 3 are the ones that matter.

## Testing without the camera

The link is plain serial, so a serial monitor is enough for all of it:

- Type `<K` and `<E VIKAS` into the monitor at 115200 and watch the beep and log.
- Walk the gates and watch for exactly one `ENTRY` per person — especially when
  walking through **fast**, which is what Change 3 addresses.
- Confirm the display and encoder stay responsive while someone stands in the
  doorway, which is what Change 4 addresses.

## Things to leave alone

- The `ENTRY  |  Occupancy: N  |  Total visitors: M` format — it is the contract.
- The Bridge RPC names `on_encoder` / `on_button` / `on_rfid` — the Python brick
  calls them by name.
- The menu, capacity editing, and alert behaviour. They work.

## Known limits, not bugs — don't "fix" these

- The display is one digit: occupancy above 9 shows `F`, and capacity is capped
  at 9. That is the hardware, not an oversight.
- `>P I` / `>Q` from an older spec are dead. Ignore any reference to them.

## What I need from you

1. Make Changes 1 and 2 first and show me the diff — small and reviewable.
2. Then Change 3, with the edge-transition logic as a clearly separated function
   I can reason about.
3. Then Change 4.
4. Keep the existing constant/config block style — I need to retune timing at the
   venue without hunting through the code.
5. No `delay()` in the main loop, and nothing that blocks serial reads.

Ask me before changing anything not listed above.

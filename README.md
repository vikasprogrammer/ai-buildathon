# AI Hardware Buildathon — Visitor & Occupancy Counter

Smart room-occupancy system that counts people entering and exiting through a door, and shows the count on a 7-segment display. Encoder + RFID reader hang off a WiFi-connected companion board, so nothing but power runs between the two boards.

## Hardware

- **Arduino UNO Q** — main controller. Runs the counting logic, drives the display, LED, and buzzer.
- **PCB Cupid Glyph C6 (ESP32-C6)** — companion board. Reads a rotary encoder and an MFRC522 RFID reader, and forwards events over WiFi.
- 2× IR obstacle sensor modules (entry/exit detection)
- 1× single-digit 7-segment display (common cathode)
- 1× KY-040 rotary encoder + push-button
- 1× MFRC522 RFID reader
- Red LED (capacity-full alert) + active buzzer

## Wiring

### Arduino UNO Q

| Component | Pin |
|---|---|
| IR1 OUT (outside) | D2 |
| IR2 OUT (inside) | D3 |
| 7-seg segments a–g | D4–D10 (220 Ω on common cathode) |
| Red LED | D11 (220 Ω in series) |
| Buzzer + | D12 |

### PCB Cupid Glyph C6

| Component | Glyph pin |
|---|---|
| Encoder CLK / DT / SW | D8 / D7 / D6 |
| RFID SCK / MOSI / MISO | D15 / D18 / D19 |
| RFID SS / RST | D14 / D20 |

The two boards are **not physically connected**. They communicate over your local WiFi (UDP broadcast) and Arduino Bridge RPC.

## Software layout

```
visitor_counter/          Arduino UNO Q MCU sketch (Arduino App Lab, Zephyr platform)
esp32_encoder_bridge/     PCB Cupid Glyph C6 sketch (Arduino IDE, ESP32-C6)
wifi_listener/            Python brick on the Uno Q's Linux side (Arduino App Lab)
```

### Data flow

```
Encoder / RFID  ->  Glyph broadcasts UDP  ->  Python brick receives on port 4210
Python brick    ->  Bridge.call("on_encoder" | "on_button" | "on_rfid")
MCU sketch      ->  updates counter / menu / display / alerts
```

## Features

- Two-IR direction detection: IR1 then IR2 within 1.5 s → ENTRY; the reverse → EXIT
- Occupancy shown live on the 7-segment display; `F` when the room is full
- Red LED + three-beep alert when occupancy hits the capacity threshold (default 8)
- Staff menu accessed via the encoder push-button:
  - `C` — set capacity (1–9)
  - `r` — reset occupancy and total-visitor counters
  - `E` — exit
- RFID card gating: encoder only responds after an allowed card is tapped
- Every RPC the brick delivers is echoed on the Uno Q's Serial Monitor for easy debugging

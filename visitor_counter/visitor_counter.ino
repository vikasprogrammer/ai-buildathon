/*
  Visitor & Occupancy Counter
  ---------------------------
  Two IR sensors detect the direction a person walks through a door.

  Wiring (Arduino Uno):
    IR1 (OUTSIDE sensor)
      VCC -> 5V
      GND -> GND
      OUT -> D2
    IR2 (INSIDE sensor)
      VCC -> 5V
      GND -> GND
      OUT -> D3
    Optional status LED on D13 (already on the board).

    7-segment display (common cathode, one digit):
      segment a -> D4
      segment b -> D5
      segment c -> D6
      segment d -> D7
      segment e -> D8
      segment f -> D9
      segment g -> D10
      common cathode -> GND  (put a 220 ohm resistor on each segment line)
      If your display is common-anode, set COMMON_CATHODE to false below
      and connect the common pin to 5V instead of GND.

    Capacity alert:
      Red LED  -> D11  (long leg via a 220 ohm resistor to D11, short leg to GND)
      Buzzer   -> D12  (+ to D12, - to GND; assumes an ACTIVE buzzer)

    Rotary encoder is NOT wired to the Arduino any more.
    It now hangs off an ESP32-C6 mini that talks to the Arduino over one wire.

    PCBCupid Glyph C6 -> Arduino UNO Q link (WiFi + Arduino Bridge RPC):
      NO wires between the two physical boards.
      1. The Glyph joins your WiFi and broadcasts UDP events (see the ESP32 sketch).
      2. A Python brick on the Uno Q's Linux side receives each UDP packet and
         calls one of our RPCs on the MCU using Arduino_RouterBridge.
      3. This sketch just exposes those RPCs (on_encoder, on_button, on_rfid).
      See wifi_listener/main.py for the Python brick.

    Requires the Arduino_RouterBridge library on the sketch side. On App Lab
    with the arduino:zephyr platform this is usually built-in; if not, add
    "Arduino_RouterBridge" under sketch.yaml's libraries.

    Encoder is wired to the Glyph on D6 (CLK), D7 (DT), D20 (SW), 3V3, GND.
    RFID reader (MFRC522) is also on the Glyph -- see the ESP32 sketch for wiring.

    RPCs exposed to the Python brick:
      on_button()                -- one press of the encoder push-button (toggles mode)
      on_count(int value)        -- current absolute encoder position from the Glyph
      on_rfid(String uid)        -- one card scan, uid is a hex string

  Admin mode:
    In NORMAL mode the 7-seg shows the current occupancy.
    Press the encoder button -> enter ADMIN mode.
      The display switches to showing the threshold. As the operator turns
      the encoder, the ESP32's `count` field updates and we display it as
      the new threshold.
    Press the encoder button again -> exit ADMIN mode.
      The threshold is saved as the last value that came in via on_count()
      while in admin mode.
    All admin activity is logged on the Serial Monitor tagged [ADMIN].

    See esp32_encoder_bridge.ino for the ESP32-C6 sketch.

  Logic:
    IR1 triggers first, then IR2 within TIMEOUT_MS  -> someone ENTERED
    IR2 triggers first, then IR1 within TIMEOUT_MS  -> someone EXITED
    If only one sensor fires and the other never does, we reset after TIMEOUT_MS.

  Most cheap IR obstacle modules pull the OUT line LOW when something is
  in front of them. If yours works the opposite way, flip DETECTED below.
*/

const int IR1_PIN = 2;   // outside sensor
const int IR2_PIN = 3;   // inside sensor
const int LED_PIN = 13;  // on-board LED, blinks on any count change

const int DETECTED = LOW;   // change to HIGH if your module is active-HIGH
const unsigned long TIMEOUT_MS = 1500;  // max gap between the two sensors

// 7-segment display pins, in order a, b, c, d, e, f, g
const int SEG_PINS[7] = {4, 5, 6, 7, 8, 9, 10};
const bool COMMON_CATHODE = true;  // set to false if you have common-anode

// Which segments to light up for each digit 0-9 (a,b,c,d,e,f,g)
const byte DIGIT_MAP[10][7] = {
  {1,1,1,1,1,1,0}, // 0
  {0,1,1,0,0,0,0}, // 1
  {1,1,0,1,1,0,1}, // 2
  {1,1,1,1,0,0,1}, // 3
  {0,1,1,0,0,1,1}, // 4
  {1,0,1,1,0,1,1}, // 5
  {1,0,1,1,1,1,1}, // 6
  {1,1,1,0,0,0,0}, // 7
  {1,1,1,1,1,1,1}, // 8
  {1,1,1,1,0,1,1}  // 9
};

// Pattern for the letter "F" -- shown when occupancy is above 9 (Full)
const byte LETTER_F[7] = {1,0,0,0,1,1,1};

// Capacity alert
int capacity = 8;               // default capacity; staff can change this via encoder
const int RED_LED_PIN = 11;     // red alert LED
const int BUZZER_PIN  = 12;     // active buzzer

// Encoder & RFID live on the ESP32-C6. Events reach us over WiFi -> Python brick
// -> Arduino Bridge RPC. The brick calls the on_encoder / on_button / on_rfid
// functions we register with Bridge.provide() in setup() below.
#include <Arduino_RouterBridge.h>

const int MIN_CAPACITY = 1;
const int MAX_CAPACITY = 9;     // single-digit display can only show up to 9
const unsigned long ADMIN_TIMEOUT_MS = 10000;  // auto-save & exit after inactivity

// Letter "C" -- briefly shown on the 7-seg when admin mode is entered
const byte LETTER_C[7] = {1,0,0,1,1,1,0};

enum Mode { MODE_NORMAL, MODE_ADMIN };
Mode mode = MODE_NORMAL;
int editValue = 0;                     // capacity value being edited in admin mode
unsigned long lastAdminActivityMs = 0;

// Events received from the ESP32 encoder bridge, waiting to be consumed
bool pendingBtn    = false;
int  pendingCount  = -1;      // -1 = never received; otherwise 0..9 from the Glyph
bool countUpdated  = false;   // set true whenever on_count writes a fresh value

char lastCardUid[24] = "";    // most recently scanned RFID UID (for reference)

// State machine: what we are currently waiting for
enum State { IDLE, WAIT_IR2_ENTRY, WAIT_IR1_EXIT };
State state = IDLE;

unsigned long stateStart = 0;
int occupancy = 0;
unsigned long totalVisitors = 0;  // total people who ever entered

void setup() {
  pinMode(IR1_PIN, INPUT);
  pinMode(IR2_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  // Bring up the Arduino Bridge so the Python brick can call our RPCs
  Bridge.begin();
  Bridge.provide("on_button", on_button);
  Bridge.provide("on_count",  on_count);
  Bridge.provide("on_rfid",   on_rfid);

  for (int i = 0; i < 7; i++) pinMode(SEG_PINS[i], OUTPUT);
  showDigit(occupancy);
  updateAlert(false);   // make sure alert is off at boot
  Serial.begin(9600);
  Serial.println(F("Visitor counter ready."));
  Serial.println(F("Occupancy: 0  |  Total visitors: 0"));
}

void loop() {
  bool btn = pollButton();          // fires once per encoder-button press
  int  newCount = 0;
  bool haveNewCount = pollCount(newCount);   // fires once per count change

  // IR counting always runs, regardless of mode
  handleSensors();

  switch (mode) {

    case MODE_NORMAL:
      // Button press in normal mode -> switch to admin mode
      if (btn) enterAdmin();
      break;

    case MODE_ADMIN:
      // Whenever a fresh count value arrives, treat that number as the
      // threshold-in-progress and show it on the 7-segment.
      if (haveNewCount) {
        editValue = constrain(newCount, MIN_CAPACITY, MAX_CAPACITY);
        showDigit(editValue);
        Serial.print(F("[ADMIN] Threshold being set -> "));
        Serial.println(editValue);
        lastAdminActivityMs = millis();
      }
      // Button press in admin mode -> save threshold and go back to normal
      if (btn) {
        exitAdmin(F("button pressed"));
      } else if (millis() - lastAdminActivityMs > ADMIN_TIMEOUT_MS) {
        // Safety net so we don't get stuck if the operator walks away
        exitAdmin(F("timeout"));
      }
      break;
  }
}

// Runs the IR entry/exit state machine (extracted from the old loop body).
void handleSensors() {
  bool ir1 = (digitalRead(IR1_PIN) == DETECTED);
  bool ir2 = (digitalRead(IR2_PIN) == DETECTED);

  switch (state) {

    case IDLE:
      if (ir1 && !ir2) {
        state = WAIT_IR2_ENTRY;   // outside broken first -> maybe entering
        stateStart = millis();
      } else if (ir2 && !ir1) {
        state = WAIT_IR1_EXIT;    // inside broken first -> maybe exiting
        stateStart = millis();
      }
      break;

    case WAIT_IR2_ENTRY:
      if (ir2) {
        occupancy++;
        totalVisitors++;
        report("ENTRY");
        waitUntilClear();
        state = IDLE;
      } else if (millis() - stateStart > TIMEOUT_MS) {
        state = IDLE;             // false alarm, reset
      }
      break;

    case WAIT_IR1_EXIT:
      if (ir1) {
        if (occupancy > 0) occupancy--;
        report("EXIT");
        waitUntilClear();
        state = IDLE;
      } else if (millis() - stateStart > TIMEOUT_MS) {
        state = IDLE;
      }
      break;
  }
}

// ---- RPC handlers called by the Python brick via Arduino Bridge ----
// Bridge invokes these from a background thread. We just set flags; the main
// loop drains them via pollButton() / pollCount() below.
// Each handler also prints to Serial so incoming events show up on the
// Uno Q's Serial Monitor tab.

void on_button() {
  pendingBtn = true;
  Serial.println(F("[link] on_button()"));
}

void on_count(int value) {
  pendingCount = value;
  countUpdated = true;
  Serial.print(F("[link] on_count("));
  Serial.print(value);
  Serial.println(F(")"));
}

void on_rfid(String uid) {
  char buf[24];
  strncpy(buf, uid.c_str(), sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = 0;
  Serial.print(F("[link] on_rfid("));
  Serial.print(buf);
  Serial.println(F(")"));
  onCardScanned(buf);
}

// React to an RFID card scan. For now: log it + brief LED blink + short beep.
void onCardScanned(const char* uid) {
  strncpy(lastCardUid, uid, sizeof(lastCardUid) - 1);
  lastCardUid[sizeof(lastCardUid) - 1] = 0;

  Serial.print(F("RFID scanned  UID: "));
  Serial.println(uid);

  // Non-blocking-ish feedback: single short beep + on-board LED blink
  digitalWrite(LED_PIN, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);
  delay(80);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
}

// Returns true once on each fresh press of the encoder button.
bool pollButton() {
  bool b = pendingBtn;
  pendingBtn = false;
  return b;
}

// Returns true if a fresh count value has arrived since last call.
// The value is written to *out*. Otherwise returns false and *out* is untouched.
bool pollCount(int& out) {
  if (!countUpdated) return false;
  out = pendingCount;
  countUpdated = false;
  return true;
}

// Enter admin mode. Seed editValue with the Glyph's most recent count (if
// we've received one) so the display starts at the value the operator sees
// on the physical knob. Otherwise fall back to the current capacity.
void enterAdmin() {
  mode = MODE_ADMIN;
  int seed = (pendingCount >= 0) ? pendingCount : capacity;
  editValue = constrain(seed, MIN_CAPACITY, MAX_CAPACITY);
  lastAdminActivityMs = millis();

  Serial.println();
  Serial.println(F("=================================="));
  Serial.println(F("[ADMIN] Entered admin mode."));
  Serial.print  (F("[ADMIN] Current capacity: "));
  Serial.println(capacity);
  Serial.print  (F("[ADMIN] Threshold display starting at: "));
  Serial.println(editValue);
  Serial.println(F("[ADMIN] Turn knob to adjust. Press button again to save."));
  Serial.println(F("=================================="));

  // Brief visual cue: flash "C" for a moment, then show the current value
  showPattern(LETTER_C);
  delay(400);
  showDigit(editValue);
}

// Leave admin mode: commit editValue as the new capacity and restore normal display.
void exitAdmin(const __FlashStringHelper* reason) {
  int oldCapacity = capacity;
  capacity = editValue;
  mode = MODE_NORMAL;
  showDigit(occupancy);
  updateAlert(occupancy >= capacity);

  Serial.println();
  Serial.println(F("=================================="));
  Serial.print  (F("[ADMIN] Exiting admin mode ("));
  Serial.print  (reason);
  Serial.println(F(")."));
  if (capacity != oldCapacity) {
    Serial.print  (F("[ADMIN] Capacity saved: "));
    Serial.print  (oldCapacity);
    Serial.print  (F(" -> "));
    Serial.println(capacity);
  } else {
    Serial.print  (F("[ADMIN] Capacity unchanged at "));
    Serial.println(capacity);
  }
  Serial.print  (F("[ADMIN] Occupancy is "));
  Serial.print  (occupancy);
  Serial.print  (F(" / "));
  Serial.println(capacity);
  Serial.println(F("=================================="));
  Serial.println();
}

// Blink the LED, refresh the 7-seg (only in normal mode), and print the current counts
void report(const char* event) {
  digitalWrite(LED_PIN, HIGH);
  if (mode == MODE_NORMAL) showDigit(occupancy);   // don't clobber the menu display
  bool full = (occupancy >= capacity);
  // If we just crossed into "full" on this event, beep three times.
  static bool wasFull = false;
  if (full && !wasFull) beepAlert();
  wasFull = full;
  updateAlert(full);
  Serial.print(event);
  Serial.print(F("  |  Occupancy: "));
  Serial.print(occupancy);
  Serial.print(F("  |  Total visitors: "));
  Serial.println(totalVisitors);
  if (full) Serial.println(F("  *** ROOM FULL ***"));
  delay(80);
  digitalWrite(LED_PIN, LOW);
}

// Turn the red LED on/off. Buzzer is only pulsed on transition (see beepAlert).
void updateAlert(bool on) {
  digitalWrite(RED_LED_PIN, on ? HIGH : LOW);
  if (!on) digitalWrite(BUZZER_PIN, LOW);
}

// Three short beeps -- fires once when occupancy first hits capacity.
void beepAlert() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(120);
    digitalWrite(BUZZER_PIN, LOW);
    delay(80);
  }
}

// Drive the seven segments from any 7-bit pattern (a..g).
void showPattern(const byte* pattern) {
  for (int i = 0; i < 7; i++) {
    bool on = pattern[i] == 1;
    digitalWrite(SEG_PINS[i], COMMON_CATHODE ? on : !on);
  }
}

// Show a digit 0-9, or "F" if the value is out of range.
void showDigit(int value) {
  if (value >= 0 && value <= 9) showPattern(DIGIT_MAP[value]);
  else                          showPattern(LETTER_F);
}

// Wait until both sensors are clear again so one person isn't counted twice
void waitUntilClear() {
  unsigned long t = millis();
  while (millis() - t < 2000) {   // safety cap of 2 seconds
    bool ir1 = (digitalRead(IR1_PIN) == DETECTED);
    bool ir2 = (digitalRead(IR2_PIN) == DETECTED);
    if (!ir1 && !ir2) return;
  }
}

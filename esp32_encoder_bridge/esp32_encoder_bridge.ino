#include <SPI.h>
#include <MFRC522.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// ---- WiFi / UDP ----
const char* WIFI_SSID     = "test";
const char* WIFI_PASSWORD = "test1234";
const int   WIFI_UDP_PORT = 4210;                 // must match Uno Q listener
IPAddress   broadcastIP(255, 255, 255, 255);       // broadcast to whole subnet

WiFiUDP udp;

// ---- RFID pins ----
#define SCK_PIN   15
#define MOSI_PIN  18
#define MISO_PIN  19
#define SS_PIN    14
#define RST_PIN   20

// ---- Encoder pins ----
#define ENC_SW   6
#define ENC_DT   7
#define ENC_CLK  8

#define ENC_MIN  0
#define ENC_MAX  9

MFRC522 rfid(SS_PIN, RST_PIN);
byte allowedUID[] = {0x61, 0x25, 0xF9, 0x17};

volatile int encoderCount = 0;
volatile bool accessGranted = false;

void sendUDP(const char* msg) {
  udp.beginPacket(broadcastIP, WIFI_UDP_PORT);
  udp.print(msg);
  udp.endPacket();
}

void IRAM_ATTR readEncoder() {
  if (!accessGranted) return;   // ignore ticks unless unlocked

  int dtState = digitalRead(ENC_DT);
  if (dtState != digitalRead(ENC_CLK)) {
    if (encoderCount < ENC_MAX) encoderCount++;
  } else {
    if (encoderCount > ENC_MIN) encoderCount--;
  }
}

void setupWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. IP: ");
  Serial.println(WiFi.localIP());
  udp.begin(WIFI_UDP_PORT);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);

  setupWiFi();

  SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN, SS_PIN);
  rfid.PCD_Init();
  delay(50);

  pinMode(ENC_SW, INPUT_PULLUP);
  pinMode(ENC_DT, INPUT);
  pinMode(ENC_CLK, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_CLK), readEncoder, CHANGE);

  Serial.println("Scan a tag to unlock encoder...");
}

void loop() {
  // Check for a new card
  if (rfid.PICC_IsNewCardPresent() && rfid.PICC_ReadCardSerial()) {
    String uidHex = "";
    for (byte i = 0; i < rfid.uid.size; i++) {
      char buf[3];
      sprintf(buf, "%02X", rfid.uid.uidByte[i]);
      uidHex += buf;
    }
    Serial.print("UID: ");
    Serial.println(uidHex);

    if (checkUID()) {
      accessGranted = true;
      encoderCount = 0;
      Serial.println("-> ACCESS GRANTED, encoder active");
    } else {
      accessGranted = false;
      Serial.println("-> ACCESS DENIED");
    }

    // always report the scanned UID to the Uno Q
    String msg = "U:" + uidHex;
    sendUDP(msg.c_str());

    rfid.PICC_HaltA();
    rfid.PCD_StopCrypto1();
  }

  // Only report encoder activity if unlocked
  if (accessGranted) {
    static int lastCount = 0;
    if (encoderCount != lastCount) {
      sendUDP(encoderCount > lastCount ? "+" : "-");
      Serial.print("Count: ");
      Serial.println(encoderCount);
      lastCount = encoderCount;
    }

    static bool lastSwState = HIGH;
    bool swState = digitalRead(ENC_SW);
    if (swState != lastSwState) {
      if (swState == LOW) {
        sendUDP("P");
        Serial.println("Button pressed");
      }
      lastSwState = swState;
    }
  }

  delay(5);
}

bool checkUID() {
  if (rfid.uid.size != sizeof(allowedUID)) return false;
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] != allowedUID[i]) return false;
  }
  return true;
}
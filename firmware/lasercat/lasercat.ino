#include <ESP32Servo.h>

Servo panServo;
Servo tiltServo;

const int PAN_PIN  = 4;
const int TILT_PIN = 5;

const int PAN_MIN = 10,  PAN_MAX = 170;
const int TILT_MIN = 40,  TILT_MAX = 150;

int panTarget = 90, tiltTarget = 90;
int panPos = 90,    tiltPos = 90;

int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

void setup() {
  Serial.begin(115200);
  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PAN_PIN, 500, 2400);
  tiltServo.attach(TILT_PIN, 500, 2400);
  panServo.write(panPos);
  tiltServo.write(tiltPos);
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    int comma = line.indexOf(',');
    if (comma > 0) {
      panTarget  = clampi(line.substring(0, comma).toInt(), PAN_MIN, PAN_MAX);
      tiltTarget = clampi(line.substring(comma + 1).toInt(), TILT_MIN, TILT_MAX);
    }
  }
  if (panPos < panTarget) panPos++; else if (panPos > panTarget) panPos--;
  if (tiltPos < tiltTarget) tiltPos++; else if (tiltPos > tiltTarget) tiltPos--;
  panServo.write(panPos);
  tiltServo.write(tiltPos);
  delay(8);
}
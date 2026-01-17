#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Servo.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

const int buzzerPin = 8;
const int servoPin = 9;
Servo myServo;

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);
  myServo.attach(servoPin);
  myServo.write(0);

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    for(;;);
  }
  refreshDisplay("READY", "Waiting...");
}

void loop() {
  if (Serial.available() > 0) {
    char data = Serial.read();

    if (data == '1') {
      // 1. Perform the 5-time action
      refreshDisplay("BIRD!", "ACTING...");
      for (int i = 0; i < 5; i++) {
        digitalWrite(buzzerPin, HIGH);
        myServo.write(90);
        delay(200);
        digitalWrite(buzzerPin, LOW);
        myServo.write(0);
        delay(200);
      }

      // 2. The 3-second Cooldown/Delay
      for (int seconds = 3; seconds > 0; seconds--) {
        refreshDisplay("COOLDOWN", String(seconds) + "s left");
        delay(1000);
      }
     
      refreshDisplay("READY", "Waiting...");
      // Clear buffer to ignore any '1's sent during the delay
      while(Serial.available() > 0) Serial.read();
    }
  }
}

void refreshDisplay(String title, String status) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println(title);
  display.drawFastHLine(0, 12, 128, WHITE);
  display.setCursor(0, 35);
  display.setTextSize(2);
  display.println(status);
  display.display();
}
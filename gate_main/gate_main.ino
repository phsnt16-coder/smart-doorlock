#include <ESP32Servo.h>
#include "esp_sleep.h"

Servo myServo;

int lastOpen = HIGH;
int lastClose = HIGH;
unsigned long lastPrint = 0;

// 39번 핀 LOW 상태 지속 감지용
unsigned long lowStartTime = 0;
bool lowTimerRunning = false;

void setup() {
  Serial.begin(115200);

  pinMode(4, INPUT_PULLUP);
  pinMode(5, INPUT_PULLUP);
  pinMode(39, INPUT);
  myServo.attach(15, 544, 2400);

  lastOpen  = digitalRead(4);
  lastClose = digitalRead(5);
  
  esp_sleep_enable_ext0_wakeup(GPIO_NUM_4, 0);
  esp_sleep_enable_ext1_wakeup( (1ULL << GPIO_NUM_39), ESP_EXT1_WAKEUP_ANY_HIGH );

}

void loop() {
  
  // --- 하강 에지 감지 ---
  int nowOpen = digitalRead(4);
  if (lastOpen == HIGH && nowOpen == LOW) { // 하강 에지
    myServo.write(180);
    delay(500);
    myServo.write(90); // 열기
  }
  lastOpen = nowOpen;


  // --- HIGH 레벨 감지 ---
  int PIRmode = digitalRead(39);
  if (PIRmode == HIGH) {
    Serial.println("WakeUp");
    delay(1000);

    // HIGH가 되면 LOW 카운트 초기화
    lowTimerRunning = false;
  } 
  else { 
    // LOW 상태일 때 카운트 시작
    if (!lowTimerRunning) {
      lowTimerRunning = true;
      lowStartTime = millis();
    }
    else if (millis() - lowStartTime >= 8000) {
      Serial.println("39번 핀 LOW 8초 지속 -> 딥슬립 진입");
      delay(100);
      esp_deep_sleep_start();
    }
  }

  if (Serial.available()) {
    String command = Serial.readStringUntil('\n'); // 줄바꿈까지 읽기
    command.trim(); // 공백 제거
    if (command.equalsIgnoreCase("Unlock")) { //Unlock 송신시
      myServo.write(180);
      delay(100);
      myServo.write(90); // 열기
    }
  }

  int nowClose = digitalRead(5);
  if (lastClose == HIGH && nowClose == LOW) { // 하강 에지
    delay(2000);
    myServo.write(0); 
    delay(500);
    myServo.write(90);// 닫기
    esp_deep_sleep_start();
  }
  lastClose = nowClose;

  // --- 1초마다 핀 상태 출력 ---
  unsigned long now = millis();
  if (now - lastPrint >= 1000) {
    lastPrint = now;
    Serial.printf("GPIO4: %d, GPIO5: %d\n", digitalRead(4), digitalRead(5));
  }
}


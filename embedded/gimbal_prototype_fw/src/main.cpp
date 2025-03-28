#include <Arduino.h>
#include <Adafruit_BNO08x.h>
#include <iq_module_communication.hpp>
#include <esp_now.h>
#include <WiFi.h>

IqSerial ser(Serial1);
PowerMonitorClient power(0);
MultiTurnAngleControlClient angle(0);

uint8_t estop_mac_addr[] = {0x30, 0x30, 0xF9, 0x34, 0x52, 0xA0};


String receivedString = "";
char send_str[255];

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) { // len should not be longer than 250bytes
    receivedString = String((char *)incomingData);
}

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {

}




void setup() {
    // Initialize the IqSerial object
    // ser.begin();
    setCpuFrequencyMhz(80);


    Serial1.begin(115200, SERIAL_8N1, D1, D0);
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(D2, OUTPUT);
    digitalWrite(D2, LOW); // ground reference for IQ motor

    
    Serial.begin(115200);
    delay(1000);

    // ESP NOW SETUP
    WiFi.mode(WIFI_MODE_STA);
    Serial.println(WiFi.macAddress());
    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        delay(3000);
        ESP.restart();
    }
    esp_now_register_recv_cb(OnDataRecv);
    esp_now_register_send_cb(OnDataSent);
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, estop_mac_addr, 6);
    peerInfo.channel = 0;
    peerInfo.encrypt = false;
    esp_now_add_peer(&peerInfo);
}

// int f0 = 4096; //ranges from 0 to 8192, 4096 is center
// int m0 = 0; // 0: voltage control, 1: angle control

int cmdx = 0;
int cmdy = 0;
int run0 = 0;
long dt = 0;


void loop() {


    float battery_voltage = 0;
    ser.get(power.volts_, battery_voltage);
    
    // if (m0 == 0) {
    //     // voltage control
    //     float ctrl_voltage = (f0 - 4096) / 4096.0f * 7.4f;
    //     ser.set(angle.ctrl_volts_, ctrl_voltage);
    // }else{
    //     // angle control
    //     float angle_deg = (f0 - 4096) / 4096.0f * 2*PI;
    //     ser.set(angle.ctrl_angle_, angle_deg);
    // }

    // // ESP NOW receive and send
    // if (receivedString.indexOf("f0") != -1) { // data exists
    //     f0 = receivedString.substring(receivedString.indexOf("f0") + 3, receivedString.indexOf("f0") + 8).toInt();
    //     m0 = receivedString.substring(receivedString.indexOf("m0") + 3, receivedString.indexOf("m0") + 8).toInt();
    // }

    
    // size_t send_str_size = sprintf(send_str,
    //     "v:%.2f\n"
    //     ,
    //     battery_voltage
    // );
    // esp_now_send(estop_mac_addr, (uint8_t *) send_str, send_str_size);


    // ESP NOW receive and send
    if (receivedString.indexOf("cmdx") != -1) { // data exists
        cmdx = receivedString.substring(receivedString.indexOf("cmdx") + 5, receivedString.indexOf("cmdx") + 10).toInt();
        cmdy = receivedString.substring(receivedString.indexOf("cmdy") + 5, receivedString.indexOf("cmdy") + 10).toInt();
        run0 = receivedString.substring(receivedString.indexOf("run0") + 5, receivedString.indexOf("run0") + 10).toInt();
    }
    size_t send_str_size = sprintf(send_str,
        "v:%.2f\n"
        ,
        battery_voltage
    );
    esp_now_send(estop_mac_addr, (uint8_t *) send_str, send_str_size);

    // Serial.printf("v:%.2f\n", battery_voltage);
    // // Serial.printf("f0:%d m0:%d\n", f0, m0);

    Serial.printf("dt: %d\n", dt);
    // Serial.printf("status: %d\n", sensorValue.status);
    // Serial.printf("yaw: %f\n", ypr.yaw);
    // Serial.printf("pitch: %f\n", ypr.pitch);
    // Serial.printf("roll: %f\n", ypr.roll);
    Serial.printf("voltage: %f\n", battery_voltage);
    Serial.printf("cmdx: %d\n", cmdx);
    Serial.printf("cmdy: %d\n", cmdy);
    Serial.printf("run0: %d\n", run0);
    Serial.printf("\t\n");

    delay(1);
}
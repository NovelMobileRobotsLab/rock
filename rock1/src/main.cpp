#include <Arduino.h>
#include <iq_module_communication.hpp>
#include <esp_now.h>
#include <WiFi.h>

// Create our IqSerial instance on Serial1
IqSerial ser(Serial1);


// Create the client we need in order to read the module's input voltage
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


    
    // Initialize Serial (for displaying information on the terminal)
    Serial.begin(115200);

    delay(1000);
    Serial.println("Hellow!!");

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

int f0 = 4096; //ranges from 0 to 8192
void loop() {
    // Serial.println(receivedString);


    float battery_voltage = 0;
    ser.get(power.volts_, battery_voltage);

    if (receivedString.indexOf("f0") != -1) { // data exists
        f0 = receivedString.substring(receivedString.indexOf("f0") + 3, receivedString.indexOf("f0") + 8).toInt();
    }

    float ctrl_voltage = (f0 - 4096) / 4096.0f * 7.4f;
    // float ctrl_voltage = 0;
    ser.set(angle.ctrl_volts_, ctrl_voltage);
    // Serial.println(f0);

    // digitalWrite(LED_BUILTIN, HIGH);
    // delay(100);
    // digitalWrite(LED_BUILTIN, LOW);
    // delay(100);



    size_t send_str_size = sprintf(send_str,
        "v:%.2f\n"
        ,
        battery_voltage
    );
    esp_now_send(estop_mac_addr, (uint8_t *) send_str, send_str_size);

    delay(1);

}
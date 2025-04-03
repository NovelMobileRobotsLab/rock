// #include <Arduino.h>
// #include <Adafruit_BNO08x.h>
// #include <iq_module_communication.hpp>
// #include <WiFi.h>
// #include <esp_now.h>

// #define BNO08X_CS 6
// #define BNO08X_INT 5
// #define BNO08X_RESET 1
// // struct euler_t {
// //     float yaw;
// //     float pitch;
// //     float roll;
// // } ypr;
// // Adafruit_BNO08x bno08x(BNO08X_RESET);
// // sh2_SensorValue_t sensorValue;
// // sh2_SensorId_t reportType = SH2_ARVR_STABILIZED_RV;
// // long reportIntervalUs = 5000;
// // void setReports(sh2_SensorId_t reportType, long report_interval) {
// //     Serial.println("Setting desired reports");
// //     if (!bno08x.enableReport(reportType, report_interval)) {
// //         Serial.println("Could not enable stabilized remote vector");
// //     }
// // }

// IqSerial ser(Serial1);
// PowerMonitorClient power(0);
// MultiTurnAngleControlClient mot(0);

// uint8_t estop_mac_addr[] = {0x30, 0x30, 0xF9, 0x34, 0x52, 0xA0};
// String receivedString = "";
// char send_str[255];

// void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) { // len should not be longer than 250bytes
//     receivedString = String((char *)incomingData);
// }

// void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
// }

// void setup(void) {
//     setCpuFrequencyMhz(80);

//     Serial.begin(115200);
//     Serial1.begin(115200, SERIAL_8N1, D7, D6);
//     delay(1000);


//     // IMU init
//     // if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT)) {
//     //     Serial.println("Failed to find BNO08x chip");
//     //     while (1) {
//     //         delay(10);
//     //     }
//     // }
//     // Serial.println("BNO08x Found!");
//     // setReports(reportType, reportIntervalUs);
//     // Serial.println("Reading events");


//     // ESP NOW SETUP
//     WiFi.mode(WIFI_MODE_STA);
//     Serial.println(WiFi.macAddress());
//     if (esp_now_init() != ESP_OK) {
//         Serial.println("Error initializing ESP-NOW");
//         delay(3000);
//         ESP.restart();
//     }
//     esp_now_register_recv_cb(OnDataRecv);
//     esp_now_register_send_cb(OnDataSent);
//     esp_now_peer_info_t peerInfo = {};
//     memcpy(peerInfo.peer_addr, estop_mac_addr, 6);
//     peerInfo.channel = 0;
//     peerInfo.encrypt = false;
//     esp_now_add_peer(&peerInfo);

//     delay(100);
// }

// // void quaternionToEuler(float qr, float qi, float qj, float qk, euler_t *ypr, bool degrees = false) {

// //     float sqr = sq(qr);
// //     float sqi = sq(qi);
// //     float sqj = sq(qj);
// //     float sqk = sq(qk);

// //     ypr->yaw = atan2(2.0 * (qi * qj + qk * qr), (sqi - sqj - sqk + sqr));
// //     ypr->pitch = asin(-2.0 * (qi * qk - qj * qr) / (sqi + sqj + sqk + sqr));
// //     ypr->roll = atan2(2.0 * (qj * qk + qi * qr), (-sqi - sqj + sqk + sqr));

// //     if (degrees) {
// //         ypr->yaw *= RAD_TO_DEG;
// //         ypr->pitch *= RAD_TO_DEG;
// //         ypr->roll *= RAD_TO_DEG;
// //     }
// // }

// // void quaternionToEulerRV(sh2_RotationVectorWAcc_t *rotational_vector, euler_t *ypr, bool degrees = false) {
// //     quaternionToEuler(rotational_vector->real, rotational_vector->i, rotational_vector->j, rotational_vector->k, ypr, degrees);
// // }

// // void quaternionToEulerGI(sh2_GyroIntegratedRV_t *rotational_vector, euler_t *ypr, bool degrees = false) {
// //     quaternionToEuler(rotational_vector->real, rotational_vector->i, rotational_vector->j, rotational_vector->k, ypr, degrees);
// // }

// int cmdx = 0;
// int cmdy = 0;
// int run0 = 0;
// long dt = 0;

// void loop() {

//     float battery_voltage = 0;
//     ser.get(power.volts_, battery_voltage);
//     // float angle_rad = ypr.yaw * DEG_TO_RAD;
//     // ser.set(mot.ctrl_angle_, angle_rad);s
//     // ser.set(mot.ctrl_volts_, angle_rad);


//     // if (bno08x.wasReset()) {
//     //     Serial.print("sensor was reset ");
//     //     setReports(reportType, reportIntervalUs);
//     // }

//     // if (bno08x.getSensorEvent(&sensorValue)) {
//     //     // in this demo only one report type will be received depending on FAST_MODE define (above)
//     //     switch (sensorValue.sensorId) {
//     //     case SH2_ARVR_STABILIZED_RV:
//     //         quaternionToEulerRV(&sensorValue.un.arvrStabilizedRV, &ypr, true);
//     //     case SH2_GYRO_INTEGRATED_RV:
//     //         // faster (more noise?)
//     //         quaternionToEulerGI(&sensorValue.un.gyroIntegratedRV, &ypr, true);
//     //         break;
//     //     }
//     //     static long last = 0;
//     //     long now = micros();
//     //     dt = now - last;
//     //     last = now;
//     // }

//     // ESP NOW receive and send
//     if (receivedString.indexOf("cmdx") != -1) { // data exists
//         cmdx = receivedString.substring(receivedString.indexOf("cmdx") + 5, receivedString.indexOf("cmdx") + 10).toInt();
//         cmdy = receivedString.substring(receivedString.indexOf("cmdy") + 5, receivedString.indexOf("cmdy") + 10).toInt();
//         run0 = receivedString.substring(receivedString.indexOf("run0") + 5, receivedString.indexOf("run0") + 10).toInt();
//     }
//     size_t send_str_size = sprintf(send_str,
//         "v:%.2f\n"
//         ,
//         battery_voltage
//     );
//     esp_now_send(estop_mac_addr, (uint8_t *) send_str, send_str_size);



//     Serial.printf("dt: %d\n", dt);
//     // Serial.printf("status: %d\n", sensorValue.status);
//     // Serial.printf("yaw: %f\n", ypr.yaw);
//     // Serial.printf("pitch: %f\n", ypr.pitch);
//     // Serial.printf("roll: %f\n", ypr.roll);
//     Serial.printf("voltage: %f\n", battery_voltage);
//     Serial.printf("cmdx: %d\n", cmdx);
//     Serial.printf("cmdy: %d\n", cmdy);
//     Serial.printf("run0: %d\n", run0);
//     Serial.printf("\t\n");

    

//     delay(1);
// }




#include <Arduino.h>
#include <Adafruit_BNO08x.h>
#include <iq_module_communication.hpp>
#include <esp_now.h>
#include <WiFi.h>

#include <model2150.h>
#include <all_ops_resolver.h>
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#define TAG "main"
#define INPUT_SIZE 48



#define BNO08X_CS 6
#define BNO08X_INT 5
#define BNO08X_RESET 1
struct euler_t {
    float yaw;
    float pitch;
    float roll;
} ypr;
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;
sh2_SensorId_t reportType = SH2_ARVR_STABILIZED_RV;
long reportIntervalUs = 5000;
void setReports(sh2_SensorId_t reportType, long report_interval) {
    Serial.println("Setting desired reports");
    if (!bno08x.enableReport(reportType, report_interval)) {
        Serial.println("Could not enable stabilized remote vector");
    }
}

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



//tensorflow
tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
// An area of memory to use for input, output, and intermediate arrays.
const int kTensorArenaSize = 35 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

void printModelInfo(){
  input = interpreter->input(0);
  ESP_LOGI(TAG, "Input Shape");
  for (int i = 0; i < input->dims->size; i++)
  {
    ESP_LOGI(TAG, "%d", input->dims->data[i]);
  }

  ESP_LOGI(TAG, "Input Type: %s", TfLiteTypeGetName(input->type));
  ESP_LOGI(TAG, "Output Shape");

  TfLiteTensor *output = interpreter->output(0);
  for (int i = 0; i < output->dims->size; i++)
  {
    ESP_LOGI(TAG, "%d", output->dims->data[i]);
  }
  ESP_LOGI(TAG, "Output Type: %s", TfLiteTypeGetName(output->type));

  ESP_LOGI(TAG, "Arena Size:%d bytes of memory", interpreter->arena_used_bytes());
}




void setup() {
    // Initialize the IqSerial object
    // ser.begin();
    setCpuFrequencyMhz(80);


    Serial1.begin(115200, SERIAL_8N1, D7, D6);
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

    

    //     // IMU init
//     // if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT)) {
//     //     Serial.println("Failed to find BNO08x chip");
//     //     while (1) {
//     //         delay(10);
//     //     }
//     // }
//     // Serial.println("BNO08x Found!");
//     // setReports(reportType, reportIntervalUs);
//     // Serial.println("Reading events");




    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    // model = tflite::GetModel(mnist_model);
    // model = tflite::GetModel(sin_model);
    // model = tflite::GetModel(sin_model_512);
    model = tflite::GetModel(model2150);
    if (model->version() != TFLITE_SCHEMA_VERSION){
        ESP_LOGE(TAG, "Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    CREATE_ALL_OPS_RESOLVER(op_resolver)

    static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk){
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return;
    }
    printModelInfo();


}


float input_scale = 0.12579839;
int input_zero_pt = 3;
float output_scale = 0.48779645562171936;
int output_zero_pt = -29;

int cmdx = 4096;
int cmdy = 4096;
int run0 = 0;
long dt = 0;

float ctrl_volt_filt = 0;

float obs[INPUT_SIZE] = {0};
float new_obs[16] = {0};

float action = 0;

void loop() {

    new_obs[0] = action; // last action

    new_obs[1] = 0; // quat[0]
    new_obs[2] = 0; // quat[1]
    new_obs[3] = 0; // quat[2]
    new_obs[4] = 0; // quat[3]

    new_obs[5] = 0; // angvel[0]/6
    new_obs[6] = 0; // angvel[1]/6
    new_obs[7] = 0; // angvel[2]/6 (rad/s)

    new_obs[8] = 0; // linvel[0]*10
    new_obs[9] = 0; // linvel[0]*10
    new_obs[10] = 0; // linvel[0]*10 (m/s)

    new_obs[11] = 0; // dofvel/50 (rad/s)
    new_obs[12] = 0; // dofpos/10 (radians)

    new_obs[13] = 0; // command[0]
    new_obs[14] = 0; // command[1]
    new_obs[15] = 0; // command[2]

    //shift observations one up
    for (int i = 1; i < INPUT_SIZE; i++){
        obs[i] = obs[i-1]; 
    }
    for (int i = 0; i < INPUT_SIZE; i+=3){
        obs[i] = new_obs[i/3];
    }


    // Εvaluate neural net
    TfLiteTensor *input = interpreter->input(0);
    for (int i = 0; i < INPUT_SIZE; i++){
        input->data.int8[i] = int8_t(constrain(round(obs[i] / input_scale + input_zero_pt), -128, 127));
    }
    long start = micros();
    if (interpreter->Invoke() == kTfLiteOk){
        Serial.printf("Invoke in: %d microseconds\n", micros() - start);
    }else{
        Serial.printf("Invoke failed!\n");
        return;
    }
    int8_t output_int = interpreter->output(0)->data.int8[0];
    float output_float = constrain((output_int - output_zero_pt) * output_scale, -1.0f, 1.0f);



    // Communicate with motor
    float battery_voltage = 0;
    ser.get(power.volts_, battery_voltage);


    if (run0 == 1) {
        float volt_alpha = 0.1;
        ctrl_volt_filt = (1-volt_alpha) * ctrl_volt_filt + (volt_alpha) * output_float * 12.0f;
        // float angle_deg = (cmdx - 4096) / 4096.0f * 2*PI;
        // ser.set(angle.ctrl_angle_, angle_deg);
        ser.set(angle.ctrl_volts_, ctrl_volt_filt);
    }else{
        // float angle_deg = (cmdx - 4096) / 4096.0f * 2*PI;
        // ser.set(angle.ctrl_angle_, angle_deg);
        float ctrl_vel = (cmdy - 4096) / 4096.0f * 50.0f;
        if (abs(ctrl_vel) < 5.0f) {
            ctrl_vel = 0.0f;
        }
        ser.set(angle.ctrl_velocity_, ctrl_vel);
    }
    


    // ESP NOW receive and send
    if (receivedString.indexOf("cmdx") != -1) { // data exists
        cmdx = receivedString.substring(receivedString.indexOf("cmdx") + 5, receivedString.indexOf("cmdx") + 10).toInt();
        cmdy = receivedString.substring(receivedString.indexOf("cmdy") + 5, receivedString.indexOf("cmdy") + 10).toInt();
        run0 = receivedString.substring(receivedString.indexOf("run0") + 5, receivedString.indexOf("run0") + 10).toInt();
    }
    size_t send_str_size = sprintf(send_str,
        "v:%.2f\n"
        "out:%f\n"
        ,
        battery_voltage,
        output_float
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
    Serial.printf("output: %f\n", output_float);
    Serial.printf("\t\n");

    delay(1);
}
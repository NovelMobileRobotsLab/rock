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

#include "util.h"


#define BNO08X_CS 6
#define BNO08X_INT 5
#define BNO08X_RESET 1




Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;
sh2_SensorId_t reportType = SH2_ARVR_STABILIZED_RV;
// sh2_SensorId_t reportType = SH2_GYRO_INTEGRATED_RV; //faster?

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
CoilTemperatureEstimatorClient coil(0);

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


float motor_zero = 0.266f; //radians

void setup() {
    delay(1000);

    setCpuFrequencyMhz(80);
    pinMode(LED_BUILTIN, OUTPUT);

    // Start Vertiq communication
    Serial1.begin(115200, SERIAL_8N1, D2, D3);
    // Serial1.begin(115200, SERIAL_8N1, D1, D0); //old rock
    pinMode(D1, OUTPUT);
    digitalWrite(D1, LOW); // ground reference for IQ motor
    delay(500);
    // ser.set(angle.sample_zero_angle_); //zero motor
    // ser.set(angle.obs_angular_displacement_, 0.0f); //zero motor
    
    Serial.begin(115200);

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

    // IMU init
    if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT)) {
        Serial.println("Failed to find BNO08x chip");
        while (1) {
            delay(10);
        }
    }
    Serial.println("BNO08x Found!");
    setReports(reportType, reportIntervalUs);
    Serial.println("Reading events");


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
int leftx = 4096;
int lefty = 4096;
int run0 = 0;
long dt = 0;

float ctrl_volt_filt = 0;

float battery_lpf = 0.1;
float battery_filt = 0;

float obs[INPUT_SIZE] = {0};
float new_obs[16] = {0};

float action = 0;


float u_origin[3] = {0,0,-1}; //corresponds to motor zero position
float v_origin[3] = {0,-1,0};
float w_origin[3] = {-1,0,0};
float d[3] = {1,0,0};

void loop() {

    if (bno08x.wasReset()) {
        Serial.print("sensor was reset ");
        setReports(reportType, reportIntervalUs);
    }

    if (bno08x.getSensorEvent(&sensorValue)) {
        // in this demo only one report type will be received depending on FAST_MODE define (above)
        switch (sensorValue.sensorId) {
        case SH2_ARVR_STABILIZED_RV:
            quaternionToEulerRV(&sensorValue.un.arvrStabilizedRV, &ypr, true);
        case SH2_GYRO_INTEGRATED_RV:
            // faster (more noise?)
            quaternionToEulerGI(&sensorValue.un.gyroIntegratedRV, &ypr, true);
            break;
        }
        static long last = 0;
        long now = micros();
        dt = now - last;
        last = now;
    }

    /*

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
        // Serial.printf("Invoke in: %d microseconds\n", micros() - start);
    }else{
        Serial.printf("Invoke failed!\n");
        return;
    }
    int8_t output_int = interpreter->output(0)->data.int8[0];
    float output_float = constrain((output_int - output_zero_pt) * output_scale, -1.0f, 1.0f);
    */
    

    // Communicate with motor
    float mot_angle = 0;
    ser.get(angle.obs_angular_displacement_, mot_angle);

    // float coil_temp = 0.0f; //doesn't work
    // ser.get(coil.t_coil_, coil_temp);

    //set desired direction
    float cmd_mag = sqrt(sq(mapf8192(cmdx)) + sq(mapf8192(cmdy)));
    if(cmd_mag > 0.5){
        d[0] = mapf8192(cmdx) / cmd_mag;
        d[1] = mapf8192(cmdy) / cmd_mag;
    }


    float u[3] = {0};
    float v[3] = {0};
    float w[3] = {0}; //motor axis
    rotateVectorByQuaternion(&sensorValue.un.arvrStabilizedRV, u_origin, u);
    rotateVectorByQuaternion(&sensorValue.un.arvrStabilizedRV, v_origin, v);
    rotateVectorByQuaternion(&sensorValue.un.arvrStabilizedRV, w_origin, w);
    float proj_angle = atan2f(-u[1]*d[0] + u[0]*d[1], v[1]*d[0]-v[0]*d[1]);
    if(w[2]>0){
        proj_angle += PI;
    }
    proj_angle = proj_angle - motor_zero;
    proj_angle = proj_angle + 2*PI*round((mot_angle - proj_angle) / (2*PI));


    float battery_voltage = -1;
    ser.get(power.volts_, battery_voltage);
    battery_filt = (1-battery_lpf)*battery_filt + battery_lpf*battery_voltage;


    if (battery_filt > 20 && run0 == 1){
        if(abs(mapf8192(lefty)) > 0.2){
            ser.set(angle.ctrl_volts_, mapf8192(lefty) * 12.0f);
        }else if(cmd_mag > 0.5){
            ser.set(angle.ctrl_angle_, proj_angle);
        }else{
            ser.set(angle.ctrl_volts_, 0.0f);
        }
    }else{
        ser.set(angle.ctrl_volts_, 0.0f);
    }


    // ESP NOW receive and send
    if (receivedString.indexOf("cmdx") != -1) { // data exists
        cmdx = receivedString.substring(receivedString.indexOf("cmdx") + 5, receivedString.indexOf("cmdx") + 10).toInt();
        cmdy = receivedString.substring(receivedString.indexOf("cmdy") + 5, receivedString.indexOf("cmdy") + 10).toInt();
        leftx = receivedString.substring(receivedString.indexOf("leftx") + 6, receivedString.indexOf("leftx") + 11).toInt();
        lefty = receivedString.substring(receivedString.indexOf("lefty") + 6, receivedString.indexOf("lefty") + 11).toInt();
        run0 = receivedString.substring(receivedString.indexOf("run0") + 5, receivedString.indexOf("run0") + 10).toInt();
    }
    size_t send_str_size = sprintf(send_str,
        "v:%.2f\n"
        ,
        battery_voltage
    );
    esp_now_send(estop_mac_addr, (uint8_t *) send_str, send_str_size);



    Serial.printf("dt: %d\n", dt);
    // Serial.printf("status: %d\n", sensorValue.status);
    Serial.printf("ypr: %f %f %f\n", ypr.yaw, ypr.pitch, ypr.roll);
    Serial.printf("voltage: %f\n", battery_filt);
    Serial.printf("mot_angle: %f\n", mot_angle * RAD_TO_DEG);
    Serial.printf("u: %f %f %f\n", u[0], u[1], u[2]);
    Serial.printf("v: %f %f %f\n", v[0], v[1], v[2]);
    Serial.printf("proj: %f\n", proj_angle * RAD_TO_DEG);
    Serial.printf("d: %f %f\n", d[0], d[1]);
    // Serial.printf("temp: %f\n", coil_temp);
    // Serial.printf("cmdy: %d\n", cmdy);
    // Serial.printf("run0: %d\n", run0);
    // Serial.printf("output: %f\n", output_float);
    Serial.printf("\t\n");

    delay(5);
}
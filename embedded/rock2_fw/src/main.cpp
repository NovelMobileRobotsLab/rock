#include <Arduino.h>
#include <Adafruit_BNO08x.h>
#include <iq_module_communication.hpp>
#include <esp_now.h>
#include <WiFi.h>

#include <model_58_02_200.h>
#include <model_13_26_700.h>

#include <vector>
#include <base64.hpp>

#include <all_ops_resolver.h>
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "util.h"

#define TAG "main"
#define INPUT_SIZE 15*3
#define BNO08X_CS 6
#define BNO08X_INT 5
#define BNO08X_RESET 1

// //58_02_200
// #define input_scale 0.007843082770705223f
// #define input_zero_pt -1
// #define output_scale 0.01251873467117548f
// #define output_zero_pt -14

// 13_26_700
#define input_scale 0.00784307811409235f
#define input_zero_pt 0
#define output_scale 0.028471535071730614f
#define output_zero_pt -12

/*
    0: spiral rock
    1: potato rock
    2: faceless rock
*/
#define ROCK_ID 2


/*
motor_zero: what motor reports when pendulum is pointing in the direction of IMU position

origin vectors are the pendulum U,V,W axes in the IMU frame
U: direction pendulum is pointing at motor zero position
V: on pendulum plane, find using right hand rule
W: motor axis pointing outwards

q_imu_to_global: {x, y, z, w} static quaternion to transform sensor quat_imu to global frame
*/
#if ROCK_ID == 0 // spiral rock
    float motor_zero = -0.05235987755f; //radians
    float u_origin[3] = {0,-1,0};
    float v_origin[3] = {0,0,-1};
    float w_origin[3] = {1,0,0};
    float q_imu_to_global[4] = {0, 0.7071068, 0.7071068, 0}; //z axis inverted, x and y swap without negating
#elif ROCK_ID == 1 // potato rock
    float motor_zero = -0.628319; //radians
    float u_origin[3] = {-1,0,0}; 
    float v_origin[3] = {0,0,-1}; 
    float w_origin[3] = {0,-1,0}; 
    float q_imu_to_global[4] = {0, 0.7071068, -0.7071068, 0}; //untested
#elif ROCK_ID == 2 // faceless rock
    float motor_zero = -2.0769418; //radians
    float u_origin[3] = {0,1,0}; 
    float v_origin[3] = {-1,0,0}; 
    float w_origin[3] = {0,0,1}; 
    float q_imu_to_global[4] = {0, 0.7071068, -0.7071068, 0}; //untested
#endif


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




void setup() {
    delay(1000);

    setCpuFrequencyMhz(240);
    pinMode(LED_BUILTIN, OUTPUT);

    // Start Vertiq communication
    #if ROCK_ID == 0
        Serial1.begin(921600, SERIAL_8N1, D2, D3);
    #else
        Serial1.begin(115200, SERIAL_8N1, D2, D3); //old rock
    #endif

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

    // model = tflite::GetModel(model_58_02_200);
    model = tflite::GetModel(model_13_26_700);
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

    // ser.set(angle.angle_Kp_, 5.0f);
    ser.set(angle.angle_Kp_, 10.0f);
    
    ser.set(angle.angle_Kd_, 0.15f);
    // ser.set(angle.angle_Kd_, 2.0f);


}



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



float d[2] = {0, 1}; //desired direction initialized to forward
float quat_imu[4]; //orientation quaternion
float angvel_imu[3]; //in IMU frame
float angvel_urdf[3]; //transformed to simulation frame

float vel_volt_ctrl_filt = 0;
float vel_volt_ctrl_lpf = 0.9378f; //0.012 looptime


void loop() {
    //get dt
    static long last = 0;
    long now = micros();
    dt = now - last;
    last = now;

    //set desired direction from joystick
    float cmd_mag = sqrt(sq(mapf8192(cmdx)) + sq(mapf8192(cmdy))); //cmdx and cmdy are read from esp-now
    if(cmd_mag > 0.5){
        d[0] = mapf8192(cmdx) / cmd_mag;
        d[1] = mapf8192(cmdy) / cmd_mag;
    }

    // Read data from motor
    float mot_angle, mot_angvel, battery_voltage = 0;
    ser.get(angle.obs_angular_displacement_, mot_angle);
    mot_angle = mot_angle - motor_zero;
    ser.get(angle.obs_angular_velocity_, mot_angvel);
    ser.get(power.volts_, battery_voltage);
    battery_filt = (1-battery_lpf)*battery_filt + battery_lpf*battery_voltage;

    long time_mot_read = micros() - last;
    last = micros();

    // Read IMU
    if (bno08x.wasReset()) {
        Serial.print("sensor was reset ");
        setReports(reportType, reportIntervalUs);
    }
    if (bno08x.getSensorEvent(&sensorValue)) {
        quaternionToEulerRV(&sensorValue.un.arvrStabilizedRV, &ypr, true); //get roll pitch yaw angles
        // quaternionToEulerGI(&sensorValue.un.gyroIntegratedRV, &ypr, true); // faster (more noise?)

        quat_imu[0] = sensorValue.un.arvrStabilizedRV.real;
        quat_imu[1] = sensorValue.un.arvrStabilizedRV.i;
        quat_imu[2] = sensorValue.un.arvrStabilizedRV.j;
        quat_imu[3] = sensorValue.un.arvrStabilizedRV.k;

        angvel_imu[0] = sensorValue.un.gyroscope.x;
        angvel_imu[1] = sensorValue.un.gyroscope.y;
        angvel_imu[2] = sensorValue.un.gyroscope.z;
    }

    long time_imu_read = micros() - last;
    last = micros();

    
    //transform IMU angular velocities to URDF frame by rotating 180º about X axis then -90º about Z axis
    angvel_urdf[0] = angvel_imu[1];
    angvel_urdf[1] = angvel_imu[0];
    angvel_urdf[2] = -angvel_imu[2];    

    // transform IMU orientation quaternion to URDF frame by multiplying it by transformation quaternion
    float quat_urdf[4];
    rotateQuaternionbyQuaternion(quat_imu, q_imu_to_global, quat_urdf);

    // Compute U,V,W of pendulum in global frame by rotating by IMU quaternion
    float u[3]; //where pendulum points at motorangle=0, towards IMU
    float v[3]; //perpendicular to u, on pendulum circle plane, y-axis of URDF 
    float w[3]; //motor axis out of motor
    rotateVectorByQuaternion(&sensorValue.un.arvrStabilizedRV, u_origin, u); //writes to u
    rotateVectorByQuaternion(&sensorValue.un.arvrStabilizedRV, v_origin, v);
    rotateVectorByQuaternion(&sensorValue.un.arvrStabilizedRV, w_origin, w);

    float thetaD = mapf8192(cmdy)*PI;
    float asymmetry = mapf8192(leftx)*PI/2.0; // angle value added to thetaM


    float uxy = sqrt(u[0]*u[0] + u[1]*u[1]);
    if(v[2]> 0){
        uxy = -uxy;
    }

    float thetaM = (atan2f(u[2], uxy) - (thetaD - PI*0.5));
    thetaM += asymmetry*u[2];
    float thetaM_motor = thetaM ; //find closest rotation to proj_angle
    thetaM_motor = thetaM_motor + motor_zero + 2*PI*round((mot_angle - thetaM_motor) / (2*PI)); //find closest rotation to proj_angle

    float proj_angle = atan2f(-u[1]*d[0] + u[0]*d[1], v[1]*d[0]-v[0]*d[1]);
    if(w[2]<0){
        proj_angle += PI;
    }
    float proj_angle_to_motor = proj_angle + 2*PI*round((mot_angle - proj_angle) / (2*PI)); //find closest rotation to proj_angle


    //fill observations vector
    new_obs[0] = action; // last action

    new_obs[1] = quat_imu[0]; // quat_imu[0] orientation in URDF frame
    new_obs[2] = quat_imu[1]; // quat_imu[1]
    new_obs[3] = quat_imu[2]; // quat_imu[2]
    new_obs[4] = quat_imu[3]; // quat_imu[3]

    new_obs[5] = angvel_urdf[0]/12.0f; // x-axis of URDF, pointing towards seeeduino, away from IMU
    new_obs[6] = angvel_urdf[1]/24.0f; // y-axis of URDF, rolls on this axis usually
    new_obs[7] = angvel_urdf[2]/12.0f; // z-axis of URDF, points down from IMU perspective

    new_obs[8] = mot_angvel / 37.5f; // dofvel/50 (rad/s)
    new_obs[9] = sin(mot_angle); // sin(mot_angle)
    new_obs[10] = cos(mot_angle); // cos(mot_angle)

    new_obs[11] = d[0]; // command[0]
    new_obs[12] = d[1]; // command[1]

    new_obs[13] = sin(-proj_angle);
    new_obs[14] = cos(proj_angle);

    //shift observation history one later
    for (int i = INPUT_SIZE-1; i > 0; i--){
        obs[i] = obs[i-1];
    }
    for (int i = 0; i < INPUT_SIZE; i+=3){
        obs[i] = new_obs[i/3];
    }

    // Εvaluate neural net
    TfLiteTensor *input = interpreter->input(0);
    for (int i = 0; i < INPUT_SIZE; i++){
        // input->data.int8[i] = int8_t(constrain(round(obs[i] / input_scale + input_zero_pt), -128, 127));
        input->data.int8[i] = int8_t(round(obs[i] / input_scale + input_zero_pt));
    }
    long start = micros();
    if (interpreter->Invoke() == kTfLiteOk){
        // Serial.printf("Invoke in: %d microseconds\n", micros() - start);
    }else{
        Serial.printf("Invoke failed!\n");
        return;
    }
    int8_t output_int = interpreter->output(0)->data.int8[0];
    action = (output_int - output_zero_pt) * output_scale;
    long time_nn = micros() - last;
    last = micros();

    float mot_vel_des = constrain(action, -1.0f, 1.0f) * 21.0f;
    float vel_volt_ctrl = constrain((mot_vel_des - mot_angvel) * 2, -15.0f, 15.0f);

    vel_volt_ctrl_filt = (1-vel_volt_ctrl_lpf)*vel_volt_ctrl_filt + vel_volt_ctrl_lpf*vel_volt_ctrl;
    

    if (battery_filt > 20 && run0 > 0){
        if(run0 == 3 && cmd_mag > 0.5){                             //run neural net
            // ser.set(angle.ctrl_velocity_, mot_vel_des);
            ser.set(angle.ctrl_volts_, vel_volt_ctrl_filt);
        }else if(cmd_mag > 0.05){                                    //manual angle projection control
            
            //ser.set(angle.ctrl_angle_, proj_angle_to_motor);
            ser.set(angle.ctrl_angle_, thetaM_motor); 
            // float kp = 4.0f;
            // float v_proportional = constrain(kp*(proj_angle_to_motor - mot_angle), -12.0f, 12.0f);
            // ser.set(angle.ctrl_volts_, v_proportional);

        }else if(abs(mapf8192(lefty)) > 0.2){                       //manual voltage control
            ser.set(angle.ctrl_volts_, mapf8192(lefty) * 12.0f);
        }else{
            ser.set(angle.ctrl_volts_, 0.0f);
        }
    }else{
        ser.set(angle.ctrl_volts_, 0.0f);
    }

    long time_mot_set = micros() - last;
    last = micros();



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
    // esp_now_send(estop_mac_addr, (uint8_t *) send_str, send_str_size);

    long time_espnow = micros() - last;

    std::vector<float> telemetry;
    telemetry.push_back(battery_filt);
    telemetry.push_back(mot_angle);
    telemetry.push_back(mot_angvel);
    telemetry.push_back(thetaM_motor); //des motor angle
    telemetry.push_back(quat_imu[0]);
    telemetry.push_back(quat_imu[1]);
    telemetry.push_back(quat_imu[2]);
    telemetry.push_back(quat_imu[3]);
    telemetry.push_back(angvel_imu[0]);
    telemetry.push_back(angvel_imu[1]);
    telemetry.push_back(angvel_imu[2]);

    size_t in_len_bytes = telemetry.size() * sizeof(float); //  How many bytes of raw float data?
    const unsigned char* in_ptr = reinterpret_cast<const unsigned char*>(telemetry.data());
    unsigned int max_out_len = static_cast<unsigned int>(((in_len_bytes + 2) / 3) * 4 + 1); //Compute a safe upper‑bound on Base64 output length
    std::unique_ptr<unsigned char[]> out_buf(new unsigned char[max_out_len]); //Allocate the output buffer
    unsigned int encoded_len = encode_base64(in_ptr, static_cast<unsigned int>(in_len_bytes), out_buf.get()); // Call encoder
    const char* payload = reinterpret_cast<const char*>(out_buf.get());
    esp_now_send(estop_mac_addr, (uint8_t *) payload, encoded_len);

    // Serial.printf("dt: %d\n", dt);
    // Serial.printf("status: %d\n", sensorValue.status);
    // Serial.printf("ypr: %f %f %f\n", ypr.yaw, ypr.pitch, ypr.roll);
    Serial.printf("voltage: %f\n", battery_filt);
    Serial.printf("mot_angle: %f\n", mot_angle * RAD_TO_DEG);
    // Serial.printf("mot_angle_zeroed: %f\n", (mot_angle+motor_zero) * RAD_TO_DEG);
    Serial.printf("u: %f %f %f\n", u[0], u[1], u[2]);
    Serial.printf("v: %f %f %f\n", v[0], v[1], v[2]);
    // Serial.printf("w: %f %f %f\n", w[0], w[1], w[2]);
    // Serial.printf("proj: %f\n", proj_angle * RAD_TO_DEG);
    Serial.printf("d: %f %f\n", d[0], d[1]);
    // Serial.printf("outint: %d\n", output_int);
    Serial.printf("vel_des: %f\n", vel_volt_ctrl_filt);


    Serial.printf("thetaM: %f\n", thetaM * RAD_TO_DEG);
    Serial.printf("thetaMtoMotor: %f\n", thetaM_motor * RAD_TO_DEG);

    Serial.printf("telemetry: %s\n", payload);


    static long time_print = 0;

    // Serial.printf("time_mot_read: %d\n", time_mot_read);
    // Serial.printf("time_imu_read: %d\n", time_imu_read);
    // Serial.printf("time_nn: %d\n", time_nn);
    // Serial.printf("time_mot_set: %d\n", time_mot_set);
    // Serial.printf("time_espnow: %d\n", time_espnow);
    // Serial.printf("time_print: %d\n", time_print);
    Serial.printf("total: %d\n", time_mot_read+time_imu_read+time_nn+time_mot_set+time_espnow+time_print);
    // Serial.printf("kp: %f\n", kp);
    // Serial.printf("kd: %f\n", kd);
    time_print = micros() - last;
    last = micros();

    // Serial.printf("oint: %d\n", output_int);

    // for (int i = 0; i < INPUT_SIZE; i++){
    //     // Serial.printf("%f ", obs[i]);
    //     Serial.printf("%d ", input->data.int8[i]);
    // }
    // Serial.printf("\n");



    // Serial.printf("cmdx: %d\n", cmdx);
    // Serial.printf("temp: %f\n", coil_temp);
    // Serial.printf("cmdy: %d\n", cmdy);
    // Serial.printf("run0: %d\n", run0);
    // Serial.printf("output: %f\n", output_float);
    Serial.printf("\t\n");

    // delay(5);
}
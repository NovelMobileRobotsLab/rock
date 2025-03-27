#include <Adafruit_BNO08x.h>
#include <Arduino.h>
#include <iq_module_communication.hpp>

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
MultiTurnAngleControlClient mot(0);

void setup(void) {

    Serial.begin(115200);
    Serial1.begin(115200, SERIAL_8N1, D7, D6);

    // Try to initialize!
    // if (!bno08x.begin_I2C()) {
    // if (!bno08x.begin_UART(&Serial1)) {  // Requires a device with > 300 byte UART buffer!
    if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT)) {
        Serial.println("Failed to find BNO08x chip");
        while (1) {
            delay(10);
        }
    }
    Serial.println("BNO08x Found!");

    setReports(reportType, reportIntervalUs);

    Serial.println("Reading events");
    delay(100);
}

void quaternionToEuler(float qr, float qi, float qj, float qk, euler_t *ypr, bool degrees = false) {

    float sqr = sq(qr);
    float sqi = sq(qi);
    float sqj = sq(qj);
    float sqk = sq(qk);

    ypr->yaw = atan2(2.0 * (qi * qj + qk * qr), (sqi - sqj - sqk + sqr));
    ypr->pitch = asin(-2.0 * (qi * qk - qj * qr) / (sqi + sqj + sqk + sqr));
    ypr->roll = atan2(2.0 * (qj * qk + qi * qr), (-sqi - sqj + sqk + sqr));

    if (degrees) {
        ypr->yaw *= RAD_TO_DEG;
        ypr->pitch *= RAD_TO_DEG;
        ypr->roll *= RAD_TO_DEG;
    }
}

void quaternionToEulerRV(sh2_RotationVectorWAcc_t *rotational_vector, euler_t *ypr, bool degrees = false) {
    quaternionToEuler(rotational_vector->real, rotational_vector->i, rotational_vector->j, rotational_vector->k, ypr, degrees);
}

void quaternionToEulerGI(sh2_GyroIntegratedRV_t *rotational_vector, euler_t *ypr, bool degrees = false) {
    quaternionToEuler(rotational_vector->real, rotational_vector->i, rotational_vector->j, rotational_vector->k, ypr, degrees);
}

void loop() {

    float battery_voltage = 0;
    ser.get(power.volts_, battery_voltage);
    float angle_rad = ypr.yaw * DEG_TO_RAD;
    // ser.set(mot.ctrl_angle_, angle_rad);s
    ser.set(mot.ctrl_volts_, angle_rad);


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
        Serial.printf("dt: %d\n", now - last);
        last = now;
        Serial.printf("status: %d\n", sensorValue.status);
        Serial.printf("yaw: %f\n", ypr.yaw);
        Serial.printf("pitch: %f\n", ypr.pitch);
        Serial.printf("roll: %f\n", ypr.roll);
        Serial.printf("voltage: %f\n", battery_voltage);
        Serial.printf("\t\n");
        // Serial.print(sensorValue.sensorId);
        // Serial.print(sensorValue.status);
        // Serial.print(""); // This is accuracy in the range of 0 to 3
        // Serial.print(ypr.yaw);
        // Serial.print("\t");
        // Serial.print(ypr.pitch);
        // Serial.print("\t");
        // Serial.println(ypr.roll);
    }

    



    delay(1);
}
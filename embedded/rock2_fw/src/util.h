#include <Arduino.h>
#include <Adafruit_BNO08x.h>


struct euler_t {
    float yaw;
    float pitch;
    float roll;
} ypr;
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

void rotateVectorByQuaternion(sh2_RotationVectorWAcc_t *quat, float v[3], float r[3]) {
    // Normalize quaternion
    float norm = sqrt(quat->real*quat->real + quat->i*quat->i + quat->j*quat->j + quat->k*quat->k);
    float qr = quat->real/norm;
    float qi = quat->i/norm;
    float qj = quat->j/norm; 
    float qk = quat->k/norm;

    // Calculate rotated vector using quaternion rotation formula:
    // v' = q * v * q^-1
    // Where q^-1 = (qr, -qi, -qj, -qk) for unit quaternions
    
    float t2 = qr*qi;
    float t3 = qr*qj;
    float t4 = qr*qk;
    float t5 = -qi*qi;
    float t6 = qi*qj;
    float t7 = qi*qk;
    float t8 = -qj*qj;
    float t9 = qj*qk;
    float t10 = -qk*qk;

    r[0] = 2*((t8 + t10)*v[0] + (t6 - t4)*v[1] + (t3 + t7)*v[2]) + v[0];
    r[1] = 2*((t4 + t6)*v[0] + (t5 + t10)*v[1] + (t9 - t2)*v[2]) + v[1];
    r[2] = 2*((t7 - t3)*v[0] + (t2 + t9)*v[1] + (t5 + t8)*v[2]) + v[2];
}

// void rotateQuaternionbyQuaternion(float q1[4], float q2[4], float result[4]){
//     // Calculate the components of the resulting quaternion

//     // {x, y, z, w}
//     // result[3] = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2];
//     // result[0] = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1];
//     // result[1] = q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0];
//     // result[2] = q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3];

//     // {w, x, y, z}
//     result[0] = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2];
//     result[1] = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1];
//     result[2] = q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0];
//     result[3] = q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3];
// }

// Multiplies two quaternions: result = q2 * q1
void rotateQuaternionbyQuaternion(const float q_sensor[4], const float q_transform[4], float q_result[4]) {
    float w1 = q_transform[0], x1 = q_transform[1], y1 = q_transform[2], z1 = q_transform[3];
    float w2 = q_sensor[0],   x2 = q_sensor[1],   y2 = q_sensor[2],   z2 = q_sensor[3];

    q_result[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    q_result[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    q_result[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    q_result[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}



float mapf(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

float mapf8192(float x) {
  return mapf(x, 0, 8192, -1, 1);
}


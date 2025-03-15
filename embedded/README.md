
# Embedded firmware for ROCK

## `convert_torch_to_tflite`
Python notebooks to convert a trained pytorch model into a .cpp and .h file that can be copied into an firmware project. Load the model using the ESP-TF Arduino library to perform inference on the ESP32S3. Will quantize the model using int8 in order to use optimized matrix operations using ESP-NN specific to the ESP32S3. 

## `gimbal_prototype_fw` 
Platformio project to control the "rock1", the first prototype that consists of a slightly slanted unbalanced flywheel that freely rotates on a vertically aligned gimbal. Uses the Seeeduino XIAO ESP32S3 and Vertiq 23-06 220KV brushless motor module.

## `esp-tf_speedtest_fw`
Platformio project to test inference speed of a fully connected neural network using the ESP-TF Arduino library. The model is trained and converted using the jupyter notebooks in `convert_torch_to_tflite`, which generates the .cpp and .h file. Uses the Seeeduino XIAO ESP32S3 and requires no hardware connections.
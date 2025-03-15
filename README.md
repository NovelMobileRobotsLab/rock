# Rolling One-motor Controlled rocK (ROCK)


<p float="middle">
  <img src="media/rockbanner.png" width="49%" />
</p>

A fully enclosed shape that tumbles and jumps.
https://miro.com/app/board/uXjVLCX5D08=/

## `embedded`

### `convert_torch_to_tflite`
Python notebooks to convert a trained pytorch model into a .cpp and .h file that can be copied into an firmware project. Load the model using the ESP-TF Arduino library to perform inference on the ESP32S3. Will quantize the model using int8 in order to use optimized matrix operations using ESP-NN specific to the ESP32S3. 

### `gimbal_prototype_fw` 
Platformio project to control the "rock1", the first prototype that consists of a slightly slanted unbalanced flywheel that freely rotates on a vertically aligned gimbal. Uses the Seeeduino XIAO ESP32S3 and Vertiq 23-06 220KV brushless motor module.

### `esp-tf_speedtest_fw`
Platformio project to test inference speed of a fully connected neural network using the ESP-TF Arduino library. The model is trained and converted using the jupyter notebooks in `convert_torch_to_tflite`, which generates the .cpp and .h file. Uses the Seeeduino XIAO ESP32S3 and requires no hardware connections.

### the below is generated with AI, will update later - Chris

## `genesis-sim`
Simulation environment for the ROCK project using the Genesis physics engine.

### `balancing`
Reinforcement learning environment and training scripts for balancing behaviors. Contains environment definition, training, evaluation scripts, and visualization tools.

### `tumbling`
Reinforcement learning environment and training scripts for tumbling behaviors. Includes environment setup, training, and evaluation scripts.

## `onshape`
CAD models and designs for various ROCK prototypes.

### `rock1`
CAD files for the first ROCK prototype with a gimbal-based design.

### `pmrock`
CAD files for a pendulum-based ROCK prototype.

### `balo` and `balo2`
CAD files for balance-oriented ROCK prototypes.

## `rockstation`
Control station software for interfacing with the physical ROCK prototypes. Includes serial communication, joystick control, and telemetry handling.

## `sandbox`
Experimental code and utilities for exploring different approaches.

### `util`
Utility functions and helper scripts.

### `trajopt`
Trajectory optimization experiments.

### `littlewood_hoop`
Experiments related to the Littlewood hoop problem.

### `gpu_accel`
GPU acceleration experiments for physics simulations.

### `from_matlab`
Code converted or adapted from MATLAB implementations.

### `acrobot`
Experiments with acrobot-style control problems.


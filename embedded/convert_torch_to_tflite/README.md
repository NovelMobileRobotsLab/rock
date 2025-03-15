# PyTorch to ONNX Neural Network Demo

This repository contains a demonstration of training a PyTorch neural network to approximate the function sin(x*y) and exporting it to ONNX format.

## Description

The demo trains a fully connected neural network with configurable parameters:
- Input size: 2 (x and y coordinates)  
- Hidden layers: [128, 128] (two hidden layers with 128 neurons each)
- Output size: 1 (the function value)

After training, the model is exported to ONNX format, which allows deployment in various environments.

## Files

- `sin_model_training.ipynb`: Jupyter notebook containing the complete demo with explanations
- `train_sin_model.py`: Python script version of the notebook code

## Requirements

```
torch>=1.7.0
numpy
matplotlib
```

## Getting Started

1. Ensure you have the required dependencies installed:
   ```
   pip install torch numpy matplotlib jupyter
   ```

2. Open and run the Jupyter notebook:
   ```
   jupyter notebook sin_model_training.ipynb
   ```

3. Alternatively, run the Python script:
   ```
   python train_sin_model.py
   ```

## Customizing the Model

You can adjust the following parameters at the top of the notebook/script:

```python
# Model architecture parameters (adjustable)
INPUT_SIZE = 2
HIDDEN_LAYERS = [128, 128]
OUTPUT_SIZE = 1

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 1000
```

## ONNX Export Details

The model is exported with the following configurations:
- ONNX opset version: 12
- Dynamic batch size
- Input names: 'input'
- Output names: 'output'
``` 
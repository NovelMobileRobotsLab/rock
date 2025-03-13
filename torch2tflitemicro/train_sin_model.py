#!/usr/bin/env python3
# Section 1: Imports and Setup
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Model architecture parameters (adjustable)
INPUT_SIZE = 2
HIDDEN_LAYERS = [128, 128]
OUTPUT_SIZE = 1

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 1000

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Section 2: Data Generation
def generate_data(num_samples=1000, x_range=(-np.pi, np.pi), y_range=(-np.pi, np.pi)):
    """Generate synthetic data for training based on sin(x*y)"""
    # Generate random inputs
    x = np.random.uniform(x_range[0], x_range[1], (num_samples, 1))
    y = np.random.uniform(y_range[0], y_range[1], (num_samples, 1))
    
    # Compute targets: sin(x*y)
    targets = np.sin(x * y)
    
    # Create input features by stacking x and y
    inputs = np.hstack((x, y))
    
    # Convert to PyTorch tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    return inputs_tensor, targets_tensor

# Create datasets
train_inputs, train_targets = generate_data(num_samples=5000)
val_inputs, val_targets = generate_data(num_samples=1000)

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Section 3: Model Definition
class SinModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_layers=HIDDEN_LAYERS, output_size=OUTPUT_SIZE):
        super(SinModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Initialize model
model = SinModel().to(DEVICE)
print(model)

# Section 4: Training Loop
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Lists to store metrics
train_losses = []
val_losses = []

# Training loop
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Section 5: Model Evaluation
def evaluate_model(model, num_test_points=1000):
    """Evaluate the model on test data and visualize predictions"""
    # Generate test data
    test_inputs, test_targets = generate_data(num_samples=num_test_points)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(test_inputs.to(DEVICE)).cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((predictions - test_targets.numpy()) ** 2)
    print(f'Test MSE: {mse:.4f}')
    
    # Create a grid for visualization
    grid_size = 50
    x = np.linspace(-np.pi, np.pi, grid_size)
    y = np.linspace(-np.pi, np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Prepare inputs for the model
    grid_inputs = np.column_stack((X.flatten(), Y.flatten()))
    grid_inputs_tensor = torch.tensor(grid_inputs, dtype=torch.float32)
    
    # Make predictions on the grid
    with torch.no_grad():
        grid_predictions = model(grid_inputs_tensor.to(DEVICE)).cpu().numpy()
    
    # Reshape predictions for plotting
    Z_pred = grid_predictions.reshape(grid_size, grid_size)
    
    # Calculate true values
    Z_true = np.sin(X * Y)
    
    # Plot predictions vs true values
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # True function
    im0 = axes[0].imshow(Z_true, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', cmap='viridis')
    axes[0].set_title('True: sin(x*y)')
    plt.colorbar(im0, ax=axes[0])
    
    # Model predictions
    im1 = axes[1].imshow(Z_pred, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', cmap='viridis')
    axes[1].set_title('Model Predictions')
    plt.colorbar(im1, ax=axes[1])
    
    # Difference (error)
    im2 = axes[2].imshow(Z_true - Z_pred, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', cmap='RdBu')
    axes[2].set_title('Difference (Error)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

# Evaluate the trained model
evaluate_model(model)

# Section 6: Export to ONNX
def export_to_onnx(model, filename='sin_model.onnx'):
    """Export the trained model to ONNX format"""
    # Create a dummy input tensor with the correct shape
    dummy_input = torch.randn(1, INPUT_SIZE, device=DEVICE)
    
    # Export the model
    torch.onnx.export(
        model,                       # model being run
        dummy_input,                 # model input (or a tuple for multiple inputs)
        filename,                    # where to save the model
        export_params=True,          # store the trained parameter weights inside the model file
        opset_version=12,            # the ONNX version to export the model to
        do_constant_folding=True,    # whether to execute constant folding for optimization
        input_names=['input'],       # the model's input names
        output_names=['output'],     # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {filename}")

# Export the model to ONNX
export_to_onnx(model) 
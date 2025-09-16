# Cats vs Dogs Image Classifier (NumPy Neural Network)

This project is a simple binary image classifier that distinguishes between cats and dogs using a fully connected neural network implemented from scratch in NumPy, without any deep learning frameworks. It’s designed for learning and experimentation with fundamental concepts like forward propagation, backpropagation, and gradient descent.

# Features

Custom 4-layer neural network (3 hidden layers + output layer)

ReLU activation for hidden layers and Sigmoid for output

Gradient clipping to stabilize training

Manual implementation of forward/backward propagation and parameter updates

Visualization of predictions on test images

# Dataset

 **Assumes the dataset is organized as:**
 ```bash
 Images/
├── Cats/
│   ├── train/
│   └── test/
├── Dogs/
│   ├── train/
│   └── test/
 ```
 **Images are resized to 64x64 pixels before training.**

# How to Run

**1. Install dependencies:**
```bash
   pip install numpy pillow matplotlib

```
**2. Prepare your dataset as shown above.**

**3. Run the main Python script:**
```python
   python train_cats_dogs_numpy.py
```
**The script will:**

*. Load and preprocess images*

*. Train the neural network*

*. Show the cost over epochs*

*. Predict on the test set*

*.Visualize a few predictions with correct/incorrect labels highlighted*

# Notes

**. This project is for learning purposes. Accuracy may be low due to simple architecture and limited   preprocessing**

**Future improvements may include:**

*.Using frameworks like TensorFlow or PyTorch*

*.Adding convolutional layers for better image feature extraction*

*.Data augmentation for improved generalization*

# Example

*The output for a test image may show:*

*Green title: prediction matches true label*

*Red title: prediction is incorrect*

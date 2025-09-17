# Cats vs Dogs Image Classifier (NumPy Neural Network)

This project implements a binary image classifier that distinguishes between cats and dogs using a fully connected neural network built from scratch with NumPy, without relying on deep learning frameworks. It is designed for learning and experimentation with core concepts like forward propagation, backpropagation, and gradient descent.

# Project Structure

```bash
image_classifier/
└── neural_classifier/
    ├── __pycache__/
    ├── __init__.py
    └── image_classifier.py   # All neural network functions

Notebook/
└── image_classifier.ipynb    # Demo notebook with training, prediction, and cost plots
```

# Feature

- 4-layer fully connected network (3 hidden layers + output)

- ReLU for hidden layers, Sigmoid for output

- Gradient clipping for stable training

- Manual forward/backward propagation and parameter updates

- Predict and visualize test images with color-coded labels (green=correct, red=wrong)

- Modular design: import functions in other scripts
  
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

# Installation

**1. Install dependencies:**
```bash
   pip install numpy pillow matplotlib

```
# Requirements

- Python 3.8+

- NumPy

- Pillow (PIL)

- Matplotlib

# How to Use
 **1. Import the module**
 
 ```python
   from image_classifier.neural_classifier import image_classifier as ic
 ```
 **2. Load images**
 ```python
    X, y, label_dict = ic.loading_images("Images", "train", (64,64))
    X, y, _ = ic.loading_images("Images", "test", (64,64))
 ```
 **3. Train the model**
 ```python
    best_params, costs = ic.train(X.T, y.reshape(1,-1),
                              input_size=64*64*3,
                              hidden_1=128, hidden_2=64, hidden_3=32,
                              output_size=1,
                              epochs=1000,
                              show_cost_change=True)
 ```
   - Training performs forward propagation, backpropagation, and parameter updates.

   - You can visualize the learning curve with plt.plot(costs).

  **4. Predict and visualize results**
  
  ```python
     y_pred = ic.Classify(X.T, y.reshape(1,-1), best_params, label_dict, num_to_show=20)
  ```
 - Correct predictions are shown in green, incorrect ones in red.
   Notes
   
# Notes

- This project is for learning purposes; accuracy may be limited due to the simple architecture.

- Future improvements could include:

  - Using frameworks like TensorFlow or PyTorch

  - Adding convolutional layers for better feature extraction

  - Implementing data augmentation for improved generalization

- The above usage is just an example, the user can modify the input parameters according to their  wished

# Example Output

- A grid of test images with predicted labels:

- Green title: prediction matches true label

- Red title: prediction is incorrect

- Cost plot over epochs shows learning progress.

# Function Inputs & Outputs

| Function                                                                                                  | Inputs                                                                                                                                                    | Outputs                                                                                                                                            | Description                                                                                      |
| --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `loading_images(data_directory, mode, image_size)`                                                        | `data_directory` (str): Root folder containing class subfolders<br>`mode` (str): 'train' or 'test'<br>`image_size` (tuple): Resize images (width, height) | `X` (np.ndarray): Flattened, normalized image vectors<br>`y` (np.ndarray): Class labels<br>`label_dict` (dict): Mapping of class names to integers | Loads images, resizes, flattens, normalizes, and labels them                                     |
| `parameter_init(input_size, hidden_1, hidden_2, hidden_3, output_size)`                                   | Layer sizes (ints)                                                                                                                                        | `parameters` (dict): Initialized weights & biases                                                                                                  | Initializes weights using He/Xavier schemes and zero biases                                      |
| `relu(Z)` / `relu_derivative(Z)`                                                                          | `Z` (np.ndarray)                                                                                                                                          | Activated values / derivatives                                                                                                                     | ReLU activation function and its derivative                                                      |
| `sigmoid(Z)` / `sigmoid_derivative(y_hat)`                                                                | `Z` (np.ndarray) / `y_hat` (np.ndarray)                                                                                                                   | Sigmoid output / derivative                                                                                                                        | Sigmoid activation for output layer and its derivative                                           |
| `forward_prop(X, parameters)`                                                                             | `X` (np.ndarray): Input data<br>`parameters` (dict): Network weights & biases                                                                             | `cache` (dict): Intermediate Z & A values for each layer                                                                                           | Computes forward propagation through all layers                                                  |
| `Cost_function(Y, y_hat)`                                                                                 | `Y` (np.ndarray): True labels<br>`y_hat` (np.ndarray): Predictions                                                                                        | `cost` (float)                                                                                                                                     | Computes binary cross-entropy loss                                                               |
| `backward_prop(parameters, cache, X_train, y_train)`                                                      | `parameters` (dict)<br>`cache` (dict)<br>`X_train`, `y_train` (np.ndarray)                                                                                | `derivatives` (dict): Gradients of weights & biases                                                                                                | Computes gradients using backpropagation                                                         |
| `parameter_update(parameters, derivatives, learning_rate=1.2, Stats=False)`                               | `parameters` (dict)<br>`derivatives` (dict)<br>`learning_rate` (float)<br>`Stats` (bool)                                                                  | `parameters` (dict): Updated weights & biases                                                                                                      | Updates parameters using gradient descent                                                        |
| `train(X, y, input_size, hidden_1, hidden_2, hidden_3, output_size, epochs=1000, show_cost_change=False)` | Training data & labels, network layer sizes, epochs, and optional cost printing                                                                           | `best_params` (dict): Best network parameters<br>`costs` (list): Cost at each epoch                                                                | Trains the network with forward/backward propagation and gradient descent                        |
| `Classify(X, Y, optimum_params, label_dict, num_to_show, cols=5)`                                         | Test data & labels, trained parameters, label mapping, number of images to display, grid columns                                                          | `y_pred` (np.ndarray): Predicted labels<br>`fig` (matplotlib figure)                                                                               | Performs predictions, computes accuracy, and displays sample images with color-coded predictions |

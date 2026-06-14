Handwritten Digit Recognition with MNIST
Python TensorFlow Keras Dataset Accuracy

A deep learning project that trains a neural network to recognize handwritten digits (0–9) using the MNIST dataset and TensorFlow/Keras.

Overview
This notebook builds and trains a simple feedforward neural network to classify 28×28 grayscale images of handwritten digits. It achieves high accuracy using a two-layer dense architecture trained on 60,000 MNIST images.

Requirements
Python 3.x
TensorFlow / Keras
scikit-learn
matplotlib
Install dependencies:

pip install tensorflow scikit-learn matplotlib
Project Structure
Handwritten.ipynb   # Main notebook with model definition, training, and evaluation
Model Architecture
Input Image (28×28)
       │
  ┌────▼────┐
  │ Flatten │  → 784 neurons
  └────┬────┘
       │
  ┌────▼────────────┐
  │  Dense (128)    │  ReLU activation
  └────┬────────────┘
       │
  ┌────▼────────────┐
  │  Dense (10)     │  Softmax activation
  └────┬────────────┘
       │
  Digit (0–9)
How It Works
Load Data — Downloads the MNIST dataset (60,000 training, 10,000 test images) via keras.datasets.mnist.
Preprocess — Normalizes pixel values from [0, 255] to [0, 1].
Build Model — A Sequential neural network:
Flatten layer to convert 28×28 images to 784-element vectors
Dense(128, activation='relu') hidden layer
Dense(10, activation='softmax') output layer for 10 digit classes
Train — Compiled with Adam optimizer and sparse categorical cross-entropy loss; trained for 10 epochs with 20% validation split.
Evaluate — Predictions made on the test set and scored with accuracy_score from scikit-learn.
Visualize — Plots for training/validation loss and accuracy over epochs.
Usage
Open and run the notebook cell by cell:

jupyter notebook Handwritten.ipynb
Results
The model tracks training and validation accuracy/loss across 10 epochs. Final accuracy on the test set is computed using scikit-learn's accuracy_score. MNIST benchmarks typically achieve 97–99% accuracy with this architecture.

# Handwritten Digit Recognition with MNIST

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Sequential-D00000?logo=keras&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-green)
![Accuracy](https://img.shields.io/badge/Accuracy-~98%25-brightgreen)

A deep learning project that trains a neural network to recognize handwritten digits (0–9) using the MNIST dataset and TensorFlow/Keras.

## Overview

This notebook builds and trains a simple feedforward neural network to classify 28×28 grayscale images of handwritten digits. It achieves high accuracy using a two-layer dense architecture trained on 60,000 MNIST images.

## Requirements

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- matplotlib

Install dependencies:

```bash
pip install tensorflow scikit-learn matplotlib
```

## Project Structure

```
Handwritten.ipynb   # Main notebook with model definition, training, and evaluation
```

## Model Architecture

```
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
```

## How It Works

1. **Load Data** — Downloads the MNIST dataset (60,000 training, 10,000 test images) via `keras.datasets.mnist`.
2. **Preprocess** — Normalizes pixel values from [0, 255] to [0, 1].
3. **Build Model** — A Sequential neural network:
   - `Flatten` layer to convert 28×28 images to 784-element vectors
   - `Dense(128, activation='relu')` hidden layer
   - `Dense(10, activation='softmax')` output layer for 10 digit classes
4. **Train** — Compiled with Adam optimizer and sparse categorical cross-entropy loss; trained for 10 epochs with 20% validation split.
5. **Evaluate** — Predictions made on the test set and scored with `accuracy_score` from scikit-learn.
6. **Visualize** — Plots for training/validation loss and accuracy over epochs.

## Usage

Open and run the notebook cell by cell:

```bash
jupyter notebook Handwritten.ipynb
```

## Results

The model tracks training and validation accuracy/loss across 10 epochs. Final accuracy on the test set is computed using scikit-learn's `accuracy_score`. MNIST benchmarks typically achieve 97–99% accuracy with this architecture.

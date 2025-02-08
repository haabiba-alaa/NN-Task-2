# Bird Species Classification using Neural Networks

## Project Overview
This project implements a **multi-layer neural network** with **Back-Propagation** to classify bird species into predefined classes. The model is trained on the first 30 samples of each class and tested on the remaining 20 samples.

## Features
- **User-configurable inputs:**
  - Enter **number of hidden layers**.
  - Enter **number of neurons** in each hidden layer.
  - Set **learning rate (eta)**.
  - Define **number of epochs (m)**.
  - Choose to **include bias** (checkbox).
  - Choose activation function: **Sigmoid or Hyperbolic Tangent Sigmoid**.

- **Training & Testing Process:**
  - Each class consists of **50 samples**.
  - **Training set:** First 30 samples per class.
  - **Testing set:** Remaining 20 samples per class.
  - **Weight initialization:** Small random values.
  
- **Classification & Evaluation:**
  - Train using **Back-Propagation**.
  - Compute **confusion matrix** and **overall accuracy**.
  - Classify a **single test sample** via GUI input.

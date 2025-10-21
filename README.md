# Deep Learning Implementations for CIFAR-10

Implementation of Batch Normalization, Dropout, and Convolutional Neural Networks (CNNs) using NumPy and PyTorch for the CIFAR-10 dataset. This project is part of the coursework for CUHK-Shenzhen's CIE6006/MCE5918 Data Analytics course.

---

## Overview

This project involves implementing and understanding key components of deep neural networks from scratch. The primary tasks are:
1.  **Batch Normalization:** Implementing the forward and backward passes and analyzing its effect on model training.
2.  **Dropout:** Implementing dropout and evaluating its role as a regularizer.
3.  **Convolutional Neural Networks (CNNs):** Building a CNN from scratch to perform image classification.
4.  **PyTorch Implementation:** Utilizing the PyTorch framework to build and train a high-performance CNN on the CIFAR-10 dataset.

## Project Structure

The core of the assignment is within the `/assignment2` directory.

```
.
└───assignment2
    │   BatchNormalization.ipynb      # Experiments with Batch Norm
    │   ConvolutionalNetworks.ipynb   # CNN implementation
    │   Dropout.ipynb                 # Experiments with Dropout
    │   PyTorch.ipynb                 # CNN implementation in PyTorch
    │
    └───cs231n
        │   layers.py                 # Implementation of layer forward/backward passes
        │
        └───classifiers
                cnn.py                # CNN model architecture
                fc_net.py             # Fully-connected network architecture
```

## Setup and Usage

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- NumPy
- Matplotlib
- PyTorch

### Steps
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TitoNicolaDrugman/Deep-Learning-from-Scratch-CIFAR10.git
    cd Deep-Learning-from-Scratch-CIFAR10/
    ```

2.  **Download the dataset:**
    Navigate to the datasets directory and run the download script.
    ```bash
    cd assignment2/cs231n/datasets/
    bash get_datasets.sh
    ```

3.  **Run the notebooks:**
    Return to the `assignment2` directory and start Jupyter to run the notebooks for each part of the assignment.
    ```bash
    cd ../../
    jupyter notebook
    ```

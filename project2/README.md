## Deep Learning - Mini Project 2
EE-559 Deep learning course at EPFL


***Authors: Filippa Bång, Håvard Bjørnøy***


The objective of this project is to design a mini “deep learning framework” using only pytorch’s
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network modules.


## Files and folders:

**helpers.py**: File containing classes for framework and training function.

**test.py**: File used for training and testing to be called without variables

## Requirements

All code was tested in a VM built on a Linux Debian 9.3 “stretch”, with miniconda and PyTorch 0.3.1.


## How to train and test the classification model
1. Install correct version of PyTorch, if needed
1. ```python3 test.py```



## Description of test.py
test.py creates a disc dataset of 1000 data points uniformly sampled in $[0, 1]^2$. 
A model with 2 input nodes, 3 hidden layers with 25 nodes and 2 output nodes, are created with following modules:
Linear - ReLU - Linear - ReLU - Linear - Tanh - Linear - Tanh.
The model is trained for 300 epochs with learning rate 0.0001 with SGD and MSE-loss. The training and test error 

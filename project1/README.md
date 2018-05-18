# Deep Learning project 1
EE-559 Deep learning course at EPFL

***Authors: Filippa Bång, Håvard Bjørnøy***

The objective of this project is to train a predictor of finger movements from Electroencephalography
(EEG) recordings.

Data Set 4: of the “BCI competition II” organized in May 2003 (Benjamin Blankertz and
M¨uller, 2002).

http://www.bbci.de/competition/ii/

It is composed of 316 training recordings, and 100 test recordings, each composed of 28 EEG channels
sampled at 1khz for 0.5s.

http://www.bbci.de/competition/ii/berlin˙desc.html

The goal is to predict the laterality of upcoming finger movements (left vs. right hand) 130 ms before
key-press.

## Files and folders

**test.py**: executable that plot and test result of the best model as well as training of the same model (with sationary learning rate and batch_size)
**helpers.py**: personal code of functions like plotting, import data, data handling etc.
**models.py**: contains all the models mentioned in the report
**blablabla.pdf**: The final report of the project

## Requirements

All code was tested in a VM built on a Linux Debian 9.3 “stretch”, with miniconda and PyTorch 0.3.1.


## How to train and test the classification model
1. Install correct version of PyTorch, if needed
1. ```python3 test.py```


## Description of test.py
test.py creates a non linear dataset and associated labels.
A model with 2 input nodes, 3 hidden layers with 25 nodes and 2 output nodes, are created with following modules:             
*Linear - ReLU - Linear - ReLU - Linear - Tanh - Linear - Tanh*                                                    
The model is trained for 300 epochs with learning rate 0.0001 with SGD and MSE-loss. The training and test error 

### References
G. C. Benjamin Blankertz and K.-R. R. M¨uller. Classifying single trial eeg: Towards brain computer
interfacing. In Neural Information Processing Systems (NIPS), 2002.


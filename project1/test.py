######   DEEP LEARNING - MINIPROJECT I   ######
"""
This file contains the testing of the best model created for \
the classification of laterality fingermovement from EEG dataset

Project description:
https://documents.epfl.ch/users/f/fl/fleuret/www/dlc/dlc-miniproject-1.pdf

"""
###############################################

# IMPORTS
# external
import torch

# personal
import helpers as HL
from models import *


# import data
train_data, val_data, test_data = HL.import_data(flatten=False, one_khz=False, train_size=253)
print("train:{:0.2f}%, val{:0.2f}%, test{:0.2f}%".format((train_data.target_tensor.size()[0]/416), (val_data.target_tensor.size()[0]/416), (test_data.target_tensor.size()[0]/416)))

# plot pretrained models 
HL.plot_model_training(history=None, models=["lstm_0.4drop_200e_100hrz","lstm2_0.2drop_200e_100hrz"]) # "lstm_200e_100hrz"

# load pretrained models and test them
HL.test_models(train_data, val_data, test_data,
["linear_1500e_lr3e-5"])

#training of approximate best model
# do not expect it to be exactly the same since the optimal model was creating using different 


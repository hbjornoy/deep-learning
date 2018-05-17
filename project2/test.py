# IMPORTS
import math
import torch
from torch import FloatTensor, LongTensor, Tensor
# our own written code
import helpers as HL

### Welcoming
print('Linear, ReLU, Linear, ReLU, Linear, Tanh, Linear, Tanh')
print('300 epochs')

### Generate data
inputs, targets = HL.generate_disc_data(n=1000, seed=123)


### Split the dataset into train, validation and test set
train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = HL.split_dataset(inputs, targets, train_perc=0.7, val_perc=0.1, test_perc=0.2)


### Normalize data
mu, std = inputs.mean(), inputs.std()
train_inputs.sub_(mu).div_(std)
validation_inputs.sub_(mu).div_(std)
test_inputs.sub_(mu).div_(std)


### Create model
input_dim = 2
hidden_width = 25
output_dim = 2

model = HL.Sequential([HL.Linear(input_dim, hidden_width), HL.ReLu(), HL.Linear(hidden_width, hidden_width), HL.ReLu(), HL.Linear(hidden_width, hidden_width), HL.Tanh(), HL.Linear(hidden_width, output_dim), HL.Tanh()])

### Train model and log training and validation error
model, train_error_list, test_error_list = HL.train_model(train_inputs, train_targets, validation_inputs, validation_targets, 
                                                    model, learning_rate = 0.0001, epochs=300)

### Print final training error
print('train_error {:.02f}%'.format(train_error_list[-1]))

### Test error
HL.test_model(model, test_inputs, test_targets)

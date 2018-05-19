# Pytorch neural net models

# IMPORTS
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import helpers as HL



#Models

class linear_model(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # set seed
        torch.manual_seed(0)
        
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # flatten input_data, once feature and sequence is now one dimension
        x = HL.flatten_input_data(x)
        out = self.fc1(x)
        return out


class logistic_model(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # set seed
        torch.manual_seed(0)
        
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.smax = nn.Softmax(dim=1)

    def forward(self, x):
        # flatten input_data, once feature and sequence is now one dimension
        x = HL.flatten_input_data(x)
        out = self.fc1(x)
        out = self.smax(out)
        return out


class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_width, dropout_rate):
        super().__init__()
        # set seed
        torch.manual_seed(0)
        
        self.fc1 = nn.Linear(input_dim, hidden_width)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_width, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # flatten input_data, once feature and sequence is now one dimension
        x = HL.flatten_input_data(x)
        
        out = self.fc1(x)
        out = self.dropout(F.relu(out))
        out = self.fc2(out)
        out = self.sig(out)
        return out

    
class conv_net1(nn.Module):
    def __init__(self, input_dim, hidden_width, output_dim):
        super().__init__()
        # makes 224 features out of the time series and 24 features
        # feature making
        # set seed
        torch.manual_seed(0)
        
        self.conv1 = nn.Conv1d(input_dim, 56, kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv1d(56, 112, kernel_size=3, stride=3, padding=0)
        self.conv3 = nn.Conv1d(112, 224, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv1d(224, 224, kernel_size=2, stride=1, padding=0)
        
        #4608 input features, 64 output features (see sizing flow below)
        #self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc1 = torch.nn.Linear(224, hidden_width)
        self.r = nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_width, output_dim)
        self.smax = torch.nn.Softmax(dim=0)
    
    def forward(self, x):
        # feature making
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        
        # remove redundant dimension before classification
        out = out.view(out.shape[0],-1)
        
        # classification
        out = self.fc1(out)
        out = self.r(out)
        out = self.fc2(out)
        out = self.smax(out)
        return out


class LSTM_net(nn.Module):
    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output):
        super().__init__ ()
        # set seed
        torch.manual_seed(0)
        
        self.lstm = nn.LSTM(input_size = dim_input, hidden_size = dim_recurrent, num_layers = num_layers, batch_first=True)
        self.fc_o2y = nn.Linear(dim_recurrent, dim_output)
        
    def forward (self, input):
        # switch places for sequence and features
        input.transpose_(1,2)
        # extract features dependant on time with LSTM
        output, _ = self.lstm(input)
        # classify with linear layer
        prev_output = self.fc_o2y(F.relu(output))
        # Get the activations of all layers at the last time step
        return F.softmax(prev_output, dim=2)[:,-1,:]
    

class LSTM_basic_dropout(nn.Module):
    def __init__(self, input_dim, output_dim, recurrent_dim, num_layers=1, dropout_rate=None):
        super().__init__ ()
        # set seed
        torch.manual_seed(0)
        
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = recurrent_dim, num_layers = num_layers, dropout=dropout_rate, batch_first=True)
        self.fc_o2y = nn.Linear(recurrent_dim, output_dim)
        
    def forward (self, input):
        # switch places for sequence and features
        input.transpose_(1,2)
        output, _ = self.lstm(input)
        output = self.fc_o2y(F.relu(output))
        #print("prev_output: ", prev_output.shape)
        #print("prev_output", prev_output)
        #print("F.softmax(prev_output, dim=2)[:,-1,:]  ", F.softmax(prev_output, dim=2)[:,-1,:])
        return F.softmax(output, dim=2)[:,-1,:] # Get the activations of all layers at the last time step



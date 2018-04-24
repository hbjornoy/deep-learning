# Helper functions and classes

# IMPORTS
# Definitely allowed imports
import math
import torch
from torch import FloatTensor, LongTensor, Tensor
import numpy as np




# Data creating/handling ------------------------------------------------------

def generate_disc_data(n=1000):
    """
    generates a dataset with a uniformly sampled data in the range [0,1] in two dimensions, with labels beeing 1 inside
    a circle with radius 1/sqrt(2*pi) and labels with 1 on the inside and 0 on the outside of the circle
    
    Output:
    inputs  : nx2 dimension FloatTensor
    targets : nx1 dimension LongTensor with range [0,1] 
    """
    inputs = torch.rand(n,2)
    euclidian_distances = torch.norm((inputs - torch.Tensor([[0.5, 0.5]])).abs(), 2, 1, True)
    targets = euclidian_distances.mul(math.sqrt(2*math.pi)).sub(1).sign().sub(1).div(2).abs().long()
    return inputs, targets


def generate_linear_data(n=1000):
    """
    generates an example dataset that can be seperated linearly
    
    Output:
    inputs  : nx2 dimension FloatTensor
    targets : nx1 dimension LongTensor with range [0,1] 
    """
    inputs = torch.rand(n,2)
    targets = torch.sum(inputs, dim=1).sub(0.9).sign().sub(1).div(2).abs().long().view(-1, 1)
    return inputs, targets


# Modules ------------------------------------------------------------------------

class Module ( object ) :
    """
    Base class that other neural network modules inherit from
    """
    
    def __init__(self):
        self._author = 'HB'
    
    def forward ( self , * input ) :
        """ `forward` should get for input, and returns, a tensor or a tuple of tensors """
        raise NotImplementedError
        
    def backward ( self , * gradwrtoutput ) :
        """
        `backward` should get as input a tensor or a tuple of tensors containing the gradient of the loss 
        with respect to the module’s output, accumulate the gradient wrt the parameters, and return a 
        tensor or a tuple of tensors containing the gradient of the loss wrt the module’s input.
        """
        raise NotImplementedError
        
    def param ( self ) :
        """ 
        `param` should return a list of pairs, each composed of a parameter tensor, and a gradient tensor 
        of same size. This list should be empty for parameterless modules (e.g. ReLU). 
        """
        return []


class Linear(Module):
    """
    Layer module: Fully connected layer defined by input dimensions and output_dimensions
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = Tensor(output_dim, input_dim).normal_(0, 1e-6)
        self.bias = Tensor(output_dim).normal_(0, 1e-6)
        self.x = 0
        self.dl_dw = Tensor(self.weight.size())
        self.dl_db = Tensor(self.bias.size())
         
    def forward(self, input):
        self.x = input
        return self.weight.mv(self.x) + self.bias
    
    def backward(self, grdwrtoutput):
        # dl_dw.add_(dl_ds.view(-1,1).mm(x.view(1,-1)))
        # accumulate derivatives
        self.dl_dw.add_(grdwrtoutput.view(-1,1).mm(self.x.view(1,-1)))
        self.dl_db.add_(grdwrtoutput)
        # returning dl_dx
        return self.weight.t().mv(grdwrtoutput)
    
    def param (self):
        return [(self.weight, self.dl_dw), (self.bias, self.dl_db)]
    
    def clear_grad(self):
        raise NotImplemented
        

class Tanh(Module):
    """
    Activation module: Tanh 
    """
    
    def __init__(self):
        super().__init__()
        self.s = 0
        
    def forward(self, input):
        self.s = input
        return torch.tanh(input)
    
    def backward(self, grdwrtoutput):
        # dl_ds = dsigma(s) * dl_dx
        return 4 * ((self.s.exp() + self.s.mul(-1).exp()).pow(-2)) * grdwrtoutput
    
    def param (self):
        return [(None, None)]
    
    def clear_grad(self):
        raise NotImplemented
        
        
class ReLu(Module):
    """
    Activation module: Leaky ReLu to avoid Dying ReLU
    """
    
    def __init__(self):
        super().__init__()
        self.s = 0
        
    def forward(self, input):
        self.s = input
        relu = input.clamp(min=0)
        #print('RELU')
        #print(relu)
        return relu
    
    def backward(self, grdwrtoutput):
        #print('INPUT TO BACKWARD')
        #print(grdwrtoutput)
        gradients = grdwrtoutput.clone()
        gradients = gradients.sign().clamp(min=0.01)
        #print('GRADIENT')
        #print(gradients)
        return gradients
    
    def param (self):
        return [(None, None)]  
        
        
        
class SGD():
    """
    SGD optimizer that alters the models parameters inplace
    """
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.params = params
        self.lr = lr
    
    def step(self):
        """does a single optimization step """
        for module in self.params:
            for tup in module:
                weight, grad = tup
                if (weight is None) or (grad is None):
                    continue
                else:
                    weight.add_(-self.lr * grad)
    
    def zero_grad(self):
        """Clears the gradients in all the models parameters"""
        for module in self.params:
            for tup in module:  
                weight, grad = tup
                if (weight is None) or (grad is None):
                    continue
                else:
                    grad.zero_()
                
                
# Models -------------------------------------------------------------------------------------

class Linear_regression_model(Module):
    """
    Linear model with no hidden layers. It is an example model for testing in development
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = Linear(input_dim, output_dim)
        self.tanh = Tanh()
        
    def forward(self, input):
        out = self.fc1.forward(input)
        out = self.tanh.forward(out)
        return out
    
    def backward(self, grdwrtoutput):
        # the first grdwrtoutput will be dl_x1 from dloss()
        dl_ds1 = self.tanh.backward(grdwrtoutput) # must store s1 from forward and cumelative dl_ds
        dl_dx0 = self.fc1.backward(dl_ds1) # must store x0 from forward
        
    def param ( self ) :
        return [self.fc1.param(), self.tanh.param()]
        
    def clear_grad(self):
        raise NotImplemented
        
        
        
# 1 Layer Model ---------------------------------------------------------------------------------

class Model_Layer(Module):
    """
    One hidden layer
    """
    def __init__(self, input_dim, output_dim, hidden_width):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_width)
        self.ReLu = ReLu()
        self.fc2 = Linear(hidden_width, output_dim)
        self.tanh = Tanh()
        
    def forward(self, input):
        out = self.fc1.forward(input)
        out = self.ReLu.forward(out)
        out = self.fc2.forward(out)
        out = self.tanh.forward(out)
        return out
    
    def backward(self, grdwrtoutput):
        # the first grdwrtoutput will be the output from dloss()
        dl_ds3 = self.tanh.backward(grdwrtoutput)
        dl_dx2 = self.fc2.backward(dl_ds3)
        dl_ds1 = self.ReLu.backward(dl_dx2)
        dl_dx0 = self.fc1.backward(dl_ds1)
        
    def param ( self ) :
        return [self.fc1.param(), self.ReLu.param(), self.fc2.param(), self.tanh.param()]
        
    def clear_grad(self):
        raise NotImplemented
        
        
        
# Temporary lossfunction -----------------------------------------------------------------------
        
def loss(pred,target):
    return (pred - target.float()).pow(2).sum()

def dloss(pred,target):
    return 2*(pred - target.float())


        

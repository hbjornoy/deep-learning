# Helper functions and classes

# IMPORTS
# Definitely allowed imports
import math
from torch import FloatTensor, LongTensor, Tensor

# Not allowed import
#import numpy as np
import torch



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


def split_dataset(inputs, targets, train_perc=0.7):
    train_inputs = inputs.narrow(0, 0, math.floor(train_perc*inputs.size()[0]))
    train_targets = targets.narrow(0, 0, math.floor(train_perc*targets.size()[0]))

    test_inputs = inputs.narrow(0, math.floor(train_perc*inputs.size()[0]), inputs.size()[0]-train_inputs.size()[0])
    test_targets = targets.narrow(0, math.floor(train_perc*targets.size()[0]), targets.size()[0]-train_targets.size()[0])
    return train_inputs, train_targets, test_inputs, test_targets

def handle_output(output_function):
    if output_function == 'linear':
        pass
    
    elif output_function == 'tanh':
        pass
    
    
def convert_to_one_hot_labels(input, target):
    tmp = input.new(target.size(0), target.max() + 1).fill_(-1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp



### MODEL FOR LINEAR WITH TWO OUTPUT NODES

def train_model(train_inputs, train_targets, test_inputs, test_targets, model, learning_rate=0.001, epochs=100, debug=False):

    """
    Trains the model and returns model and train and test error
    
    ///TODO:
    - make criterion an input
    """
    
    # make train targets and test targets to 1-hot vector
    train_targets = convert_to_one_hot_labels(train_inputs, train_targets)
    test_targets = convert_to_one_hot_labels(test_inputs, test_targets)    
    
    
    # define optimizer
    sgd = SGD(model.param(), lr=learning_rate)
    
    # constants
    nb_train_samples = train_inputs.size(0)
    nb_classes = train_targets.size(1)
    input_dim = train_inputs.size(1)
    
    
    # training in epochs
    test_error_list = []
    train_error_list = []

    for epoch in range(epochs):
        
        # Training -------------------------------------------------------------------------------
        acc_loss = 0
        nb_train_errors = 0
        # iterate through samples and accumelate derivatives
        for n in range(0, nb_train_samples):
            # clear gradiants 1.(outside loop with samples = GD) 2.(inside loop with samples = SGD)
            sgd.zero_grad()
            
            ### In order to get nb_train_errors, check how many correctly classified
            
            a_train_target = train_targets[n]
            train_targets_list = [a_train_target[0], a_train_target[1]]
            correct = train_targets_list.index(max(train_targets_list)) # argmax
            
            # Find which one is predicted of the two outputs, by taking argmax
            output = model.forward(train_inputs[n])
            output_list = [output[0], output[1]]

            prediction = output_list.index(max(output_list)) # argmax
           
            
            # Check if predicted correctly
            if int(correct) != int(prediction) : nb_train_errors += 1
            

            ### Calculate loss 
            acc_loss = acc_loss + loss(output, train_targets[n].float())
            dl_dloss = dloss(output, train_targets[n].float())
  

            if debug:
                print("output: ", output)
                print("pred: ", prediction)
                print("target: ", targets[n])
                print("loss: ", loss(output, train_targets[n]))
                print("dloss: ", dloss(output, train_targets[n]))
             
            model.backward(dl_dloss)

            ### Gradient step 1.(outside loop with samples = GD) 2.(inside loop with samples = SGD)
            sgd.step()
            
        train_error_list.append((100 * nb_train_errors) / train_inputs.size(0))


        # Testing --------------------------------------------------------------------------------
        nb_test_errors = 0
        
        for n in range(0, test_inputs.size(0)):
            
            
            ### In order to get nb_train_errors, check how many correctly classified
            
            a_test_target = test_targets[n]
            test_targets_list = [a_test_target[0], a_test_target[1]]
            correct = test_targets_list.index(max(test_targets_list)) # argmax
            
            ### Find which one is predicted of the two outputs, by taking argmax            
            
            output = model.forward(test_inputs[n])
            output_list = [output[0], output[1]]
            prediction = output_list.index(max(output_list))
            if int(correct) != int(prediction) : nb_test_errors += 1

        if epoch%(epochs/10) == 0:
            print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
              .format(epoch,
                      acc_loss,
                      (100 * nb_train_errors) / train_inputs.size(0),
                      (100 * nb_test_errors) / test_inputs.size(0)))
        test_error_list.append((100 * nb_test_errors) / test_inputs.size(0))

    return model, train_error_list, test_error_list




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
    
    have epsilon as a input
    """
    def __init__(self, input_dim, output_dim, epsilon=10e-1):
        super().__init__()
        self.weight = Tensor(output_dim, input_dim).normal_(0, epsilon)
        self.bias = Tensor(output_dim).normal_(0, epsilon)
        self.x = 0
        self.dl_dw = Tensor(self.weight.size())
        self.dl_db = Tensor(self.bias.size())
         
    def forward(self, input, debug=False):
        self.x = input
        if debug:
            print("\nFORWARD:")
            print("input: ", input)
            print("weights: ", self.weight)
            print("bias: ", self.bias)
            print("output: ", (self.weight.mv(self.x) + self.bias))
        return self.weight.mv(self.x) + self.bias
    
    def backward(self, grdwrtoutput, debug=False):
        if debug:
            print("\nBACKWARD:")
            print("grdwrtoutput: ", grdwrtoutput)
            print("weights: ", self.weight)
            print("bias: ", self.bias)
            print("update dl_dw: ", grdwrtoutput.view(-1,1).mm(self.x.view(1,-1)))
            print("dl_dw: ", self.dl_dw)
            print("dl_db: ", self.dl_db)

        self.dl_dw.add_(grdwrtoutput.view(-1,1).mm(self.x.view(1,-1)))
        self.dl_db.add_(grdwrtoutput)
        # returning dl_dx
        if debug:
            print("dl_dw after: ", self.dl_dw)
            print("dl_db after: ", self.dl_db)
            print('output(self.weight.t().mv(grdwrtoutput)): ', self.weight.t().mv(grdwrtoutput))
        return self.weight.t().mv(grdwrtoutput)
    
    def param (self):
        return [(self.weight, self.dl_dw), (self.bias, self.dl_db)]
        

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

        
class ReLu(Module):
    """
    Activation module: ReLu
    """
    
    def __init__(self):
        super().__init__()
        self.s = 0
        
    def forward(self, input):
        self.s = input
        relu = input.clamp(min=0)
        return relu
    
    def backward(self, grdwrtoutput):
        gradients = grdwrtoutput.clone()
        gradients = gradients.sign().clamp(min=0)

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
        
    def forward(self, input):
        out = self.fc1.forward(input)
        return out
    
    def backward(self, grdwrtoutput):
        # the first grdwrtoutput will be dl_x1 from dloss()
        dl_dx0 = self.fc1.backward(grdwrtoutput)
        
    def param ( self ) :
        return [self.fc1.param()]
    
    
    
class Linear_Tanh_model(Module):
    """
    Linear model with a Tanh in the end
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
        dl_ds1 = self.tanh.backward(grdwrtoutput)
        dl_dx0 = self.fc1.backward(dl_ds1)

        
       
    def param ( self ) :
        return [self.fc1.param(), self.tanh.param()]
    
    
    
class Linear_TanhTanh_model(Module):
    
    """
    Linear model with a Tanh in the end
    """
    def __init__(self, input_dim, output_dim):
        hidden_width = 25
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_width)
        self.tanh1 = Tanh()
        self.fc2 = Linear(hidden_width, output_dim)
        self.tanh2 = Tanh()
        
    def forward(self, input):
        out = self.fc1.forward(input)
        out = self.tanh1.forward(out)
        out = self.fc2.forward(out)
        out = self.tanh2.forward(out)
        return out
    
    def backward(self, grdwrtoutput):
        # the first grdwrtoutput will be dl_x1 from dloss()
        dl_ds3 = self.tanh2.backward(grdwrtoutput)
        dl_dx2 = self.fc2.backward(dl_ds3)
        dl_ds1 = self.tanh1.backward(dl_dx2)
        dl_dx0 = self.fc1.backward(dl_ds1)
       
    def param ( self ) :
        return [self.fc1.param(), self.tanh1.param(), self.fc2.param(), self.tanh2.param()]
    
    
    
    
    
class Linear_Relu_model(Module):
    
    """
    Linear model with a Tanh in the end
    """
    def __init__(self, input_dim, output_dim):
        hidden_width = 25
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_width)
        self.relu = ReLu()
        self.fc2 = Linear(hidden_width, output_dim)
        self.tanh = Tanh()
        
    def forward(self, input):
        out = self.fc1.forward(input)
        out = self.relu.forward(out)
        out = self.fc2.forward(out)
        out = self.tanh.forward(out)
        return out
    
    def backward(self, grdwrtoutput):
        # the first grdwrtoutput will be dl_x1 from dloss()
        dl_ds3 = self.tanh.backward(grdwrtoutput)
        dl_dx2 = self.fc2.backward(dl_ds3)
        dl_ds1 = self.relu.backward(dl_dx2)
        dl_dx0 = self.fc1.backward(dl_ds1)
       
    def param ( self ) :
        return [self.fc1.param(), self.relu.param(), self.fc2.param(), self.tanh.param()]


        
        
# Temporary lossfunction -----------------------------------------------------------------------

def loss(pred,target):
    return (pred - target.float()).pow(2).sum()

def dloss(pred,target):
    return 2*(pred - target.float())



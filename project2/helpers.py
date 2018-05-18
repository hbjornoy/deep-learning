# IMPORTS
import math
from torch import FloatTensor, LongTensor, Tensor
import torch


# Data creation/handling ------------------------------------------------------

def generate_disc_data(n=1000):
    """
    Generates a dataset with a uniformly sampled data in the range [0,1] in two dimensions, with labels beeing 1 inside
    a circle with radius 1/sqrt(2*pi) and labels with 1 on the inside and 0 on the outside of the circle
    
    Output:
    inputs  : nx2 dimension FloatTensor
    targets : nx1 dimension LongTensor with range [0,1] 
    """
    torch.manual_seed(123)
    inputs = torch.rand(n,2)
    euclidian_distances = torch.norm((inputs - torch.Tensor([[0.5, 0.5]])).abs(), 2, 1, True)
    targets = euclidian_distances.mul(math.sqrt(2*math.pi)).sub(1).sign().sub(1).div(2).abs().long()
    return inputs, targets


def generate_linear_data(n=1000):
    """
    Generates an example dataset that can be seperated linearly
    
    Output:
    inputs  : nx2 dimension FloatTensor
    targets : nx1 dimension LongTensor with range [0,1] 
    """
    torch.manual_seed(123)
    inputs = torch.rand(n,2)
    targets = torch.sum(inputs, dim=1).sub(0.9).sign().sub(1).div(2).abs().long().view(-1, 1)
    return inputs, targets


def split_dataset(inputs, targets, train_perc=0.7, val_perc=0.1, test_perc=0.2):
    """
    Splits dataset into training, validation and test set
    
    Output:
    train-, validation- and test inputs  : (percentage * n)x2 dimension FloatTensor
    train-, validation- and test targets : (percentage * n)x1 dimension LongTensor
    """
    train_len = math.floor(inputs.size()[0] * train_perc)
    val_len = math.floor(inputs.size()[0] * val_perc)
    test_len = math.floor(inputs.size()[0] * test_perc)
    
    train_inputs = inputs.narrow(0, 0, train_len)
    train_targets = targets.narrow(0, 0, train_len)
    
    validation_inputs = inputs.narrow(0, train_len, val_len)
    validation_targets = targets.narrow(0, train_len, val_len)

    test_inputs = inputs.narrow(0, train_len+val_len, test_len)
    test_targets = targets.narrow(0, train_len+val_len, test_len)
    
    return train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets

    
def convert_to_one_hot_labels(input, target):
    """
    Convertes targets to one-hot labels of -1 and 1
    
    Output:
    one_hot_labels : nx2 dimension FloatTensor 
    """
    tmp = input.new(target.size(0), target.max() + 1).fill_(-1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp


# Training and Testing of model ------------------------------------------------------

def train_model(train_inputs, train_targets, test_inputs, test_targets, model, learning_rate=0.001, epochs=100):
    """
    Trains the model, logs training- and validation error

    Output:
    model       :  Sequential object
    train error :  List object 
    test error  :  List object 
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
            
            # Get index of correct one, by taking argmax
            a_train_target = train_targets[n]
            train_targets_list = [a_train_target[0], a_train_target[1]]
            correct = train_targets_list.index(max(train_targets_list))
            
            
            output = model.forward(train_inputs[n])
            
            # Get index of the predicted of the two outputs, by taking argmax
            output_list = [output[0], output[1]]

            prediction = output_list.index(max(output_list))
            
            # Check if predicted correctly
            if int(correct) != int(prediction) : nb_train_errors += 1


            ### Calculate loss 
            acc_loss = acc_loss + loss(output, train_targets[n].float())
            dl_dloss = dloss(output, train_targets[n].float())

             
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
                

        if epoch%(epochs/20) == 0:
            print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% validation_error {:.02f}%'
              .format(epoch,
                      acc_loss,
                      (100 * nb_train_errors) / train_inputs.size(0),
                      (100 * nb_test_errors) / test_inputs.size(0)))
        test_error_list.append((100 * nb_test_errors) / test_inputs.size(0))

    return model, train_error_list, test_error_list


def test_model(model, test_inputs, test_targets):
    """
    Test the model and prints the test error
    """   
    
    # make test targets to 1-hot vector
    test_targets = convert_to_one_hot_labels(test_inputs, test_targets)    
    
    test_error_list = []
    
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


    print('test_error {:.02f}%'.format(((100 * nb_test_errors) / test_inputs.size(0))))
    test_error_list.append((100 * nb_test_errors) / test_inputs.size(0))
    return


# Modules ------------------------------------------------------------------------

class Module (object) :
    """
    Base class for other neural network modules to inherit from
    """
    
    def __init__(self):
        self._author = 'HB_FB'
    
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
        of same size. This list should be empty for parameterless modules (e.g. activation functions). 
        """
        return []


class Linear(Module):
    """
    Layer module: Fully connected layer defined by input dimensions and output_dimensions
    
    Outputs:
    forward  :  FloatTensor of size m (m: number of units)
    backward :  FloatTensor of size m (m: number of units)
    """
    def __init__(self, input_dim, output_dim, epsilon=1):
        super().__init__()
        torch.manual_seed(123)
        self.weight = Tensor(output_dim, input_dim).normal_(mean=0, std=epsilon)
        self.bias = Tensor(output_dim).normal_(0, epsilon)
        self.x = 0
        self.dl_dw = Tensor(self.weight.size())
        self.dl_db = Tensor(self.bias.size())
         
    def forward(self, input):
        self.x = input
        return self.weight.mv(self.x) + self.bias
    
    def backward(self, grdwrtoutput):
        self.dl_dw.add_(grdwrtoutput.view(-1,1).mm(self.x.view(1,-1)))
        self.dl_db.add_(grdwrtoutput)
        return self.weight.t().mv(grdwrtoutput)
    
    def param (self):
        return [(self.weight, self.dl_dw), (self.bias, self.dl_db)]
        

class Tanh(Module):
    """
    Activation module: Tanh 
    
    Outputs:
    forward  :  FloatTensor of size m (m: number of units)
    backward :  FloatTensor of size m (m: number of units)
    """
    
    def __init__(self):
        super().__init__()
        self.s = 0
        
    def forward(self, input):
        self.s = input
        
        # apply tanh elementwise to torch
        tanh_vector = []
        for x in input:
            tanh = (2/ (1 + math.exp(-2*x))) -1
            tanh_vector.append(tanh)
        tanh_vector = torch.FloatTensor(tanh_vector)
        return tanh_vector
    
    def backward(self, grdwrtoutput):
        return 4 * ((self.s.exp() + self.s.mul(-1).exp()).pow(-2)) * grdwrtoutput
    
    def param (self):
        return [(None, None)]

        
class ReLu(Module):
    """
    Activation module: ReLu
    
    Outputs:
    forward  :   FloatTensor of size m (m: number of units)
    backward :   FloatTensor of size m (m: number of units)
    """
    def __init__(self):
        super().__init__()
        self.s = 0
        
    def forward(self, input):
        self.s = input
        relu = input.clamp(min=0)
        return relu
    
    def backward(self, grdwrtoutput):
        relu_input = self.s
        der_relu = relu_input.sign().clamp(min=0)
        gradient_in = grdwrtoutput * der_relu
        return gradient_in    

    def param (self):
        return [(None, None)]  
        
        
class SGD():
    """
    SGD optimizer
    """
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.params = params
        self.lr = lr
    
    def step(self):
        """Single optimization step """
        for module in self.params:
            for tup in module:
                weight, grad = tup
                if (weight is None) or (grad is None):
                    continue
                else:
                    weight.add_(-self.lr * grad)
    
    def zero_grad(self):
        """Clears the gradients in all the modules parameters"""
        for module in self.params:
            for tup in module:  
                weight, grad = tup
                if (weight is None) or (grad is None):
                    continue
                else:
                    grad.zero_()
                


# Sequential -------------------------------------------------------------------------------    

class Sequential(Module):
    """
    A module combining several modules in basic sequential structure
    
    Outputs:
    parameters :  List object containing List objects with the parameters of the modules in the Sequential instance. 
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        args = list(args)[0]
        for ind, module in enumerate(args):
            self.add_module(str(ind), module)

    def add_module(self, ind, module):
        self.ind = module
        self.modules.append(self.ind)
        return module
    
    def forward(self, input):
        out = input
        for module in self.modules:
            out = module.forward(out)
        return out
    
    def backward(self, grdwrtoutput):
        reversed_modules = self.modules[::-1]
        out = grdwrtoutput
        for module in reversed_modules:
            out = module.backward(out)
    
    def param ( self ) :
        parameters = []
        for module in self.modules:
            parameters.append(module.param())
        return parameters

        
        
# Lossfunction -----------------------------------------------------------------------

def loss(pred,target):
    """
    Calculate loss 
    
    Outputs:
    loss :  float
    """
    return (pred - target.float()).pow(2).sum()

def dloss(pred,target):
    """
    Calculate derivative of loss 
    
    Outputs:
    derivative :  FloatTensor with same dimension as input
    """
    return 2*(pred - target.float())



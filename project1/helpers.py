# Python file with helpful functions for pytorch neural network coding

#IMPORTS
import dlc_bci as bci
import torch
from torch import Tensor, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
from math import log10





#FUNCTIONS

#Datahandling ---------------------------------------------------------------------------------------------------------------------

def import_data(flatten=False):
    train_input , train_target = bci.load(root = './ data_bci')
    print ( str ( type ( train_input ) ) , train_input.size() )
    print ( str ( type ( train_target ) ) , train_target.size() )
    test_input , test_target = bci.load(root = './ data_bci', train = False)
    print ( str ( type ( test_input ) ) , test_input.size())
    print ( str ( type ( test_target ) ) , test_target.size())

    # if flatten the "discard" the timeinformation, and just look at every feature at a certain time as a feature
    if flatten == True:
        train_input = flatten_input_data(train_input)
        test_input = flatten_input_data(test_input)

    train_data = TensorDataset(train_input, train_target)
    test_data = TensorDataset(test_input, test_target)
    
    return train_data, test_data


def flatten_input_data(input_data):
    """
    flatten the dimensions with 28 features with 50 samples per feature into 1400 "features"
    """
    return input_data.view(-1, input_data.size()[1]*input_data.size()[2])

#----------------------------------------------------------------------------------------------------------------------------------


#Training models-------------------------------------------------------------------------------------------------------------------

def train_model(train_data, test_data, model, criterion, learning_rate=0.0001,
                        epochs=10, batch_size=64, threads=2):

    # creating dataloaders to enable mini-batches
    train_data_loader = DataLoader(dataset=train_data, num_workers=threads, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(dataset=test_data, num_workers=threads, batch_size=batch_size, shuffle=False)
    print("Number of trainsamples: ",train_data.__len__())
    print("Number of testsamples: ",test_data.__len__())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    for epoch in range(epochs):
        train(train_data_loader, model, criterion, optimizer, epoch, epochs, train_data.__len__())
        test(test_data_loader, model, criterion, optimizer, epoch, epochs, test_data.__len__())
        checkpoint(model, epoch, epochs)
    
    
    
    return model

def train(train_data_loader, model, criterion, optimizer, epoch, epochs, nb_samples):
    epoch_loss = 0
    error_rate_list = list()
    for i, batch in enumerate(train_data_loader, 1):
        inputs = Variable(batch[0])
        targets = Variable(batch[1].type(torch.FloatTensor))

        optimizer.zero_grad() # clear gradiants
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        
        preds = outputs.round().abs().view(-1)
        error_rate_list.append(int(sum((preds != targets)))/ len(preds))
        #print(train_data_loader.__len__())      

        #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, i, len(train_data_loader), loss.data[0]))
    loss = epoch_loss / nb_samples
    error_rate = np.mean(error_rate_list)
    if epoch % np.floor(epochs/10) == 0:
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, loss))
        print("===> Prediciton Train-error: {:.4f}".format(error_rate))
    return [epoch, error_rate, loss]
    
    
def test(test_data_loader, model, criterion, optimizer, epoch, epochs, nb_samples):
    avg_psnr = 0
    error_rate_list = list()
    for batch in test_data_loader:
        inputs = Variable(batch[0])
        targets = Variable(batch[1].type(torch.FloatTensor))
        
        outputs = model(inputs)
        mse = criterion(outputs, targets)
        
        # calculate psnr (peak signal-to-noise ratio) as a measure of trust in results
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        # calculate prediction error
        preds = outputs.round().abs().view(-1)
        error_rate_list.append(int(sum((preds != targets))) / len(preds))
        
    error_rate = np.mean(error_rate_list)
    psnr = avg_psnr / nb_samples
    if epoch % np.floor(epochs/10) == 0:
        print("===> Avg. PSNR: {:.4f} dB".format(psnr))
        print("===> Prediciton  TEST-error: {:.4f}".format(error_rate))
    return [epoch, error_rate, psnr]
    
def checkpoint(model, epoch, epochs):
    model_out_path = "checkpoint_models/model_epoch_{}.pth".format(epoch)
    if epoch % np.floor(epochs/10) == 0:
        torch.save(model, model_out_path)
        print("Checkpoint: {} -------------------------------------".format(model_out_path))
#----------------------------------------------------------------------------------------------------------------------------------

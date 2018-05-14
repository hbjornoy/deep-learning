# Python file with helpful functions for pytorch neural network coding

#IMPORTS
import dlc_bci as bci
import torch
from torch import Tensor, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
from math import log10
import time
import matplotlib.pyplot as plt
import pickle

from models import *






#FUNCTIONS

#Datahandling ---------------------------------------------------------------------------------------------------------------------

def import_data(flatten=False, one_khz = False, train_size=270):
    train_input , train_target = bci.load(root = './ data_bci', one_khz=one_khz)
    print("Original data format: ")
    print ( str ( type ( train_input ) ) , train_input.size() )
    print ( str ( type ( train_target ) ) , train_target.size() )
    test_input , test_target = bci.load(root = './ data_bci', train = False, one_khz=one_khz)
    print ( str ( type ( test_input ) ) , test_input.size())
    print ( str ( type ( test_target ) ) , test_target.size())
    
    # split traindata into train and validation datasets
    val_input, val_target = train_input[train_size:], train_target[train_size:]
    train_input, train_target = train_input[:train_size], train_target[:train_size]

    # if flatten the "discard" the timeinformation, and just look at every feature at a certain time as a feature
    if flatten == True:
        train_input = flatten_input_data(train_input)
        val_input = flatten_input_data(val_input)
        test_input = flatten_input_data(test_input)

    train_data = TensorDataset(train_input, train_target)
    val_data = TensorDataset(val_input, val_target)
    test_data = TensorDataset(test_input, test_target)
    
    # One-hotting targets
    labels = train_data.target_tensor
    train_data.target_tensor = torch.LongTensor(labels.size(0), 2).zero_().scatter_(1, labels.view(-1, 1), 1)

    labels = val_data.target_tensor
    val_data.target_tensor = torch.LongTensor(labels.size(0), 2).zero_().scatter_(1, labels.view(-1, 1), 1)
    
    labels = test_data.target_tensor
    test_data.target_tensor = torch.LongTensor(labels.size(0), 2).zero_().scatter_(1, labels.view(-1, 1), 1)
    
    print("Modified train_data.data_tensor shape: ", train_data.data_tensor.shape)
    print("Modified train_data.target_tensor shape: ", train_data.target_tensor.shape)
    print("val_data.data_tensor shape: ", val_data.data_tensor.shape)
    return train_data, val_data, test_data


def flatten_input_data(input_data):
    """
    flatten the dimensions with 28 features with 50 samples per feature into 1400 "features"
    """
    return input_data.view(-1, input_data.size()[1]*input_data.size()[2])

#----------------------------------------------------------------------------------------------------------------------------------


#Training models-------------------------------------------------------------------------------------------------------------------

def train_model(train_data, val_data, model, criterion, optimizer, learning_rate=1e-3,
                        epochs=10, batch_size=64, threads=2, checkpoint_name='model'):

    # creating dataloaders to enable mini-batches
    train_data_loader = DataLoader(dataset=train_data, num_workers=threads, batch_size=batch_size, shuffle=False)
    val_data_loader = DataLoader(dataset=val_data, num_workers=threads, batch_size=batch_size, shuffle=False)
    print("Number of trainsamples: ",train_data.__len__())
    print("Number of valsamples: ",val_data.__len__())
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    
    history = np.ndarray((epochs, 6))
    timekeeper = np.array(epochs)
    
    start = time.time()
    for epoch in range(epochs):
        train_data = train(train_data_loader, model, criterion, optimizer, epoch, epochs, train_data.__len__())
        val_data = val(val_data_loader, model, criterion, optimizer, epoch, epochs, val_data.__len__())
        checkpoint(model, epoch, epochs, checkpoint_name)
        
        # save data from training for later plotting
        history[epoch, 0], history[epoch, 1], history[epoch, 2] = train_data[0], train_data[1], train_data[2]
        history[epoch, 3], history[epoch, 4], history[epoch, 5] = val_data[1], val_data[2], (time.time() - start)
        
    # save parameters using in training
    parameters = {"model": model, "criterion": criterion, "optimizer": optimizer, "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size, "threads": threads}
    
    return model, history, parameters


def train(train_data_loader, model, criterion, optimizer, epoch, epochs, nb_samples):
    # set model in training state
    model.train()
    
    epoch_loss = 0
    error_rate_list = list()
    for i, batch in enumerate(train_data_loader, 1):
        inputs = Variable(batch[0])
        targets = Variable(batch[1].type(torch.FloatTensor))
        optimizer.zero_grad() # clear gradients
        
        outputs = model(inputs)
        #print("outputs: ", outputs)
        #print("targets: ", targets)
        loss = criterion(outputs, targets)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        
        # get class prediction and label from one-hot encoding
        preds = outputs.max(dim=1)[1]
        labels = targets.max(dim=1)[1]
        error_rate_list.append(int(sum((preds != labels)))/ len(preds))     

    loss = epoch_loss / nb_samples
    error_rate = np.mean(error_rate_list)
    if epoch % np.floor(epochs/10) == 0:
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, loss))
        print("===> Prediction TRAIN-error: {:.4f}".format(error_rate))
    return [epoch, error_rate, loss]

    
def val(val_data_loader, model, criterion, optimizer, epoch, epochs, nb_samples):
    # set model in evaluation state to ignore dropout etc.
    model.eval()
    
    epoch_loss = 0
    error_rate_list = list()
    for batch in val_data_loader:
        inputs = Variable(batch[0])
        targets = Variable(batch[1].type(torch.FloatTensor))
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        epoch_loss += loss.data[0]
        
        # get class prediction and label from one-hot encoding
        preds = outputs.max(dim=1)[1] 
        labels = targets.max(dim=1)[1]
        error_rate_list.append(int(sum((preds != labels))) / len(preds))
        
    loss = epoch_loss / nb_samples
    error_rate = np.mean(error_rate_list)
    if epoch % np.floor(epochs/10) == 0:
        print("===> Avg. VAL-loss: {:.4f}".format(loss))
        print("===> Prediction  VAL-error: {:.4f}".format(error_rate))
    return [epoch, error_rate, loss]


def checkpoint(model, epoch, epochs, checkpoint_name):
    model_out_path = "checkpoint_models/" + checkpoint_name + "_epoch_{}.pth".format(epoch)
    if epoch % np.floor(epochs/10) == 0:
        torch.save(model, model_out_path)
        print("Checkpoint: {} -------------------------------------".format(model_out_path))
        

def test_model(test_data, model, criterion, batch_size=64, threads=2):
    # set model in evaluation state to ignore dropout etc.
    model.eval()
    test_data_loader = DataLoader(dataset=test_data, num_workers=threads, batch_size=batch_size, shuffle=False)
    
    epoch_loss = 0
    error_rate_list = list()
    
    for batch in test_data_loader:
        inputs = Variable(batch[0])
        targets = Variable(batch[1].type(torch.FloatTensor))
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        epoch_loss += loss.data[0]
        
        # get class prediction and label from one-hot encoding
        preds = outputs.max(dim=1)[1] 
        labels = targets.max(dim=1)[1]
        error_rate_list.append(int(sum((preds != labels))) / len(preds))
    
    loss = epoch_loss / test_data.__len__()
    error_rate = np.mean(error_rate_list)
    
    return error_rate, loss

def test_models(train_data, val_data, test_data, models):
    row_format ="| {:<26} " * (4)
    print("\nCalculating prediction error on an ensemble of trained models\n")
    print(row_format.format("Model name:", "Train error:", "Validation error:", "Test error"))
    for model_name in models:
        model_dict = unpickle_model(model_name)
        p = model_dict["parameters"]
        train_error_rate, train_loss = test_model(train_data, p["model"], p["criterion"], p["batch_size"], p["threads"])
        val_error_rate, val_loss = test_model(val_data, p["model"], p["criterion"], p["batch_size"], p["threads"])
        test_error_rate, test_loss = test_model(test_data, p["model"], p["criterion"], p["batch_size"], p["threads"])
        print(row_format.format(model_name, train_error_rate, val_error_rate, test_error_rate))

        
    
#-------------------------------------------------------------------------------------------------------------------------------


#Plotting -------------------------------------------------------------------------------------------------------------------

def plot_model_training(history, models=["lstm_1"], name="NEW"): # ex: model=["lstm", "dense"]
    fig, axarr = plt.subplots(2, 2, figsize=(15,10))
    # store all the data about from training for all the models in histories
    histories = list()
    if models is not None:
        for model_name in models:
            model_dict = unpickle_model(model_name)
            histories.append(model_dict["history"])
    if history is not None:
        histories.append(history)
        models.append(name)
    
    for history, name in zip(histories, models):
        # plot accuracy
        create_subplot(axarr[0,0], history[:,0], [history[:,1],history[:,3]], 
                       labels=[name+'_train_acc', name+'_val_acc'], ylabel="prediction error", xlabel="epochs")

        # plot loss
        create_subplot(axarr[0,1], history[:,0], [history[:,2],history[:,4]], 
                       labels=[name+'_train_loss', name+'_val_loss'], ylabel="loss")

        # plot with respect to time (only interesting with several models)
        create_subplot(axarr[1,0], history[:,5], [history[:,1],history[:,3]], 
                       labels=[name+'_train_acc', name+'_val_acc'], ylabel="prediction error", xlabel="time (seconds)")

    plt.show()
    
def create_subplot(ax, x, y_s, labels, ylabel, xlabel="epochs"):
    # plot accuracy
    ax.plot(x, y_s[0], label=labels[0])
    ax.plot(x, y_s[1], label=labels[1])
    ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
#Pickling --------------------------------------------------------------------------------------------------------------------

def pickle_model(model, history, parameters, name):
    dict_with_model_info = {"model": model, "history": history, "parameters": parameters, "name": name}
    pickle.dump(dict_with_model_info, open( "pickled_models/" + name + ".pkl", "wb" ))

def unpickle_model(name):
    return pickle.load(open("pickled_models/"+ name + ".pkl", "rb" ))


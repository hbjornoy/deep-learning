import matplotlib.pyplot as plt
from torch import Tensor, nn
import numpy as np
from math import log10
import time

from models import *
import helpers as HL

#Plotting -------------------------------------------------------------------------------------------------------------------

def plot_model_training(history, models=["lstm_1"], name="NEW", plt_name="test"): # ex: model=["lstm", "dense"]
    fig, axarr = plt.subplots(2, 2, figsize=(15,10))
    # store all the data about from training for all the models in histories
    histories = list()
    if models is not None:
        for model_name in models:
            model_dict = HL.unpickle_model(model_name)
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
    plt.savefig("images/lineplot_" + plt_name)
    plt.show()
    
def create_subplot(ax, x, y_s, labels, ylabel, xlabel="epochs"):
    # plot accuracy
    ax.plot(x, y_s[0], label=labels[0])
    ax.plot(x, y_s[1], label=labels[1])
    ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
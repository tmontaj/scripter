'''
Fit function
'''
import wandb
import tensorflow as tf
import os
from .callbacks import callbacks
from .error_rates import *
from .ctc_decoder import ctc_decoder

def fit(train_set, val_set, n_epochs, model,
        optimizer, loss, save_path, strategy, hcallbacks, restart=True):
    '''
    fit function
    Arguments:
    train_set -- train dataset
    val_set -- valedation dataset
    n_epochs -- number of epocs to train
    model -- model to train
    optimizer -- optimizer to be used
    loss -- loss function
    save_path -- path to save model
    strategy -- multi GPU/TPU strategy
    hcallbacks -- callbacks hyper parametars 
    '''
    wandb.init(project='audio2text')
    # with strategy.scope():

    if restart:
        weights= os.listdir(save_path+"/temp")
        for i in weights:
            os.system("rm -r "+save_path+"/temp"+"/"+i)
  
    else: 
        epochs = []
        weights= os.listdir(save_path+"/temp")
        if len(weights)>1:
            for i in weights:
                epochs.append = int(i.split("_")[1])
            del_ = min(epochs)
            os.system("rm -r "+save_path+"/temp/epoch_"+str(del_))
            epoch = max(epochs)
        else:
            epoch = int(weights[0].split("_")[1])

    # metrics = [LER(object_error_rate, decoder=ctc_decoder),
    #             WER(object_error_rate, decoder=ctc_decoder)]
    # metrics = [Token_ER(object_error_rate, decoder=ctc_decoder)]
    model.compile(loss=loss, optimizer=optimizer)#, metrics=metrics)
    callbacks_=callbacks(path=save_path, **hcallbacks)
    model.fit(train_set,
            validation_data = val_set,
            epochs=n_epochs,
            callbacks=callbacks_,
            steps_per_epoch=5,
            validation_steps=3
            )
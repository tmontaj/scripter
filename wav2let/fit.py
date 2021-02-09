'''
Fit function
'''
import wandb
import tensorflow as tf
from .callbacks import callbacks

def fit(train_set, val_set, n_epocs, model,
        optimizer, loss, save_path, strategy, hcallbacks):
    '''
    fit function
    Arguments:
    train_set -- train dataset
    val_set -- valedation dataset
    n_epocs -- number of epocs to train
    model -- model to train
    optimizer -- optimizer to be used
    loss -- loss function
    save_path -- path to save model
    strategy -- multi GPU/TPU strategy
    hcallbacks -- callbacks hyper parametars 
    '''
    wandb.init(project='audio2text')

    with strategy.scope():
        model.compile(loss=loss,
                        optimizer=optimizer)

    callbacks_=callbacks(path=save_path, **hcallbacks)

    model.fit(train_set,
              epochs=12,
              callbacks=callbacks_
              )

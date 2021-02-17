'''
Define callbacks for wav2letter
'''
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import os
import tensorflow as tf

# wandb
# wandb.init(project='audio2text')

# # early stopping
# patience = 2

# # Reduce on plateau
# factor = 0.1
# patience_plateau = 10

# learning rate  .. reduce exponintially after epoc 10

def scheduler(epoch, lr):
    '''
    learning rate scheduler per epoch
    '''
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# save at the end of the epoc and delete the past


class ModelPause(tf.keras.callbacks.Callback):
    '''
    Model pause (to work on spot instances)
    '''
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        path=self.path 
        self.model.save(path+"/temp/"+str(epoch)+".h5")
        files = os.listdir(path)
        if len(files) > 1:
            os.remove(path+"/temp/"+str(epoch-1)+".h5")


class ModelSave(tf.keras.callbacks.Callback):
    """
    Model save every n epoch
    """
    def __init__(self, n_epoch, path, **kwargs):
        super().__init__(**kwargs)
        self.n_epoch = n_epoch
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        path=self.path
        if epoch % self.n_epoch == 0:
            self.model.save_weights(path+"/_"+str(epoch)+".h5")

def callbacks(path, n_epoch, patience, factor, patience_plateau):
    '''
    returns list of callbacks used
    Arguments:
    path -- path to save model
    n_epoch -- number of epoch to save models(save per epoch)
    patience -- patience before early stopping
    factor -- factor to reduce learning rate on plateau
    patience_plateau -- patience to reduce learning rate on plateau

    Returns:
    out -- list of callbacks
    '''
    return [
        WandbCallback(),
        EarlyStopping(patience=patience),
        #LearningRateScheduler(schedule = scheduler),
        ModelPause(path),
        ModelSave(n_epoch, path),
        ReduceLROnPlateau(monitor='val_loss', factor=factor,
                          patience=patience_plateau)
    ]

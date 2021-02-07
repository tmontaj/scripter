import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import os

# wandb
wandb.init(project='audio2text')

# early stopping
patience = 2

# Reduce on plateau
factor=0.1
patience_plateau=10

# learning rate  .. reduce exponintially after epoc 10
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# save at the end of the epoc and delete the past 
class ModelPause(tf.keras.callbacks.Callback):
  def __init__(self,**kwargs, path = ""):
    super().__init__(**kwargs)
    self.path = path
    
  def on_epoch_end(self, epoch, logs=None):
    self.model.save_weights(path+"/"+str(epoch)+".h5")
    os.remove(path+"/"+str(epoch-1)+".h5")

class ModelSave(tf.keras.callbacks.Callback):
  def __init__(self,**kwargs, path = ""):
    super().__init__(**kwargs)
    self.path = path
    
  def on_epoch_end(self, epoch, logs=None):
    if epoch % 5 == 0:
      self.model.save_weights(path+"/_"+str(epoch)+".h5")


callbacks = [
    WandbCallback(),
    EarlyStopping(patience= patience ),
    #LearningRateScheduler(schedule = scheduler),
    ModelPause(), 
    ModelSave(),
    ReduceLROnPlateau(monitor='val_loss', factor=factor , patience= patience_plateau)
]
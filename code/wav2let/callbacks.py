import wandb
from wandb.keras import WandbCallback
import tensorflow.keras.callbacks

# wandb
wandb.init(project='audio2text')
wandb.init()

# early stopping
patience = 2

# learning rate  .. reduce exponintially after epoc 10
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# save at the end of the epoc
class ModelSave(tf.keras.callbacks.Callback):
  def __init__(self,**kwargs, path = ""):
    super().__init__(**kwargs)
    self.path = path
    
  def on_epoch_end(self, epoch, logs=None):
    self.model.save_weights(path+"/"+str(epoch)+".h5")


callbacks = [
    WandbCallback(),
    EarlyStopping(patience= patience ),
    LearningRateScheduler(schedule = scheduler),
    ModelSave(),  
     # pause and resume save
]
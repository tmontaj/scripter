# import tensorflow as tf
# from .loss import ctc_loss
# from .model import Wav2Let
# from .callbacks import callbacks

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#   model = Wav2Let()
#   model.compile(loss=ctc_loss,
#                 optimizer=tf.keras.optimizers.Adam())

# model.fit(train_dataset, epochs=12, callbacks= callbacks)
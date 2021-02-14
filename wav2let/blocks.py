'''
Building blocks (layers) of wav2letter model
'''
import tensorflow as tf

class FirstBlock(tf.keras.layers.Layer):
    '''
    First layer of wav2letter for melspectrogem (not raw audio)
    '''

    def __init__(self, filters=250, kernel_size=48, strides=2, **kwargs):
        '''
        First layer of wav2letter for melspectrogem (not raw audio)
        Arguments:
        filters -- number of filters in first conv layer(Default: 250)
        kernel_size -- kernal size in first conv layer(Default: 48)
        strides -- strides in first conv layer(Default: 2)

        **For more details see tf.keras.layers.Conv1D Docs**
        '''
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                           strides=strides, padding='same',
                                           name="first")

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, input_):
        '''
        First layer of wav2letter architecture
        Arguments:
        input_ -- input tensor

        Returns:
        out -- output tensor
        '''
        conv = self.conv(input_)
        batch_norm = self.batch_norm(conv)
        relu = self.relu(batch_norm)
        return relu

class MidBlock(tf.keras.layers.Layer):  
    '''
    Mid layers of wav2letter for melspectrogem (not raw audio)
    '''
    def __init__(self, name, filters=250, kernel_size=7, **kwargs):
        '''
        Mid layers of wav2letter
        Arguments:
        filters -- number of filters in mid conv layer(Default: 250)
        kernel_size -- kernal size in mid conv layer(Default: 48)
        name -- layer name

        **For more details see tf.keras.layers.Conv1D Docs**
        '''
        super().__init__(**kwargs)

        self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                           padding='same', name=name)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, input_):
        '''
        Mid layer of wav2letter architecture
        Arguments:
        input_ -- input tensor

        Returns:
        out -- output tensor
        '''
        conv = self.conv(input_)
        batch_norm = self.batch_norm(conv)
        relu = self.relu(batch_norm)
        return relu

class LastBlock(tf.keras.layers.Layer):
    '''
    Last layers of wav2letter for melspectrogem (not raw audio)
    '''
    def __init__(self, output_size=29, **kwargs):
        '''
        Last layers of wav2letter
        Arguments:
        output_size -- number or char in language (Default: 40)
        '''
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters=2000, kernel_size=32,
                                            padding='same', name="last_mid")

        self.conv2 = tf.keras.layers.Conv1D(filters=2000, kernel_size=1,
                                            padding='same', name="last1")

        self.conv3 = tf.keras.layers.Conv1D(filters=output_size, kernel_size=1,
                                            padding='same', name="last2")

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

    def call(self, input_):
        '''
        Last layer of wav2letter architecture
        Arguments:
        input_ -- input tensor

        Returns:
        out -- output tensor
        '''
        conv1 = self.conv1(input_)
        batch_norm1 = self.batch_norm(conv1)
        relu1 = self.relu(batch_norm1)

        conv2 = self.conv2(relu1)
        batch_norm2 = self.batch_norm2(conv2)
        relu2 = self.relu(batch_norm2)

        conv3 = self.conv3(relu2)
        batch_norm3 = self.batch_norm3(conv3)
        relu3 = self.relu(batch_norm3)

        return relu3

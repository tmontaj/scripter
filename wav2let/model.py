'''
wav2letter model
'''
from .blocks import *

class Wav2Let(tf.keras.Model):
    '''wav2letter model'''

    def __init__(self, filters=250, kernel_size=48, strides=2, depth=7,
                 mid_filters=250, mid_kernel_size=7, output_size=40, **kwargs):
        '''
        wav2letter model
        Arguments:
        filters -- number of filters in first conv layer(Default: 250)
        kernel_size -- kernal size in first conv layer(Default: 48)
        strides -- strides in first conv layer(Default: 2)
        mid_filters -- number of filters in mid conv layer(Default: 250)
        mid_kernel_size -- kernal size in mid conv layer(Default: 48)
        name -- layer name
        depth -- number od mid layers to use (Default: 7)
        output_size -- number or char in language (Default: 40)

        **For more details see tf.keras.layers.Conv1D Docs**
        '''
        super().__init__(**kwargs)
        self.first_block = FirstBlock()

        self.mid_block = []
        for i in range(depth):
            self.mid_block.append(
                MidBlock(name="mid%d" % (i))
            )

        self.last_block = LastBlock()

    def call(self, input_):
        '''
        wav2letter architecture
        Arguments:
        input_ -- input tensor

        Returns:
        out -- output tensor
        '''
        block_out = self.first_block(input_)

        for layer in self.mid_block:
            block_out = layer(block_out)

        last_block = self.last_block(block_out)

        return last_block

if __name__== "__main__":

    import numpy as np
    x= np.ones((1,15,10))
    model = Wav2Let()
    print(model(x))
    #print(model.summary())

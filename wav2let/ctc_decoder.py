import tensorflow as tf
import numpy as np

# inputs of shape (batch, time_stips, features)
def ctc_decoder(inputs, sequence_length=None, beam_width=10, top_paths=1):

  #if not padded
  if sequence_length == None:
    # print("if")
    sequence_length = [inputs.shape[1]]
    ishape = inputs.shape
    inputs = tf.reshape(inputs, (ishape[1], ishape[0], ishape[2]))
  else:
    # print("else")
    inputs = tf.stack(inputs.tolist(), axis=1)

  # print(inputs)
  decode = tf.nn.ctc_beam_search_decoder(inputs=inputs, sequence_length=sequence_length, beam_width=beam_width, top_paths=top_paths)

  return decode

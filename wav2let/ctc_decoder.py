import tensorflow as tf
import numpy as np

# inputs of shape (batch, time_stips, features)
def ctc_decoder(inputs, sequence_length=None, beam_width=10, top_paths=1):
  def ctc_decoder_(inputs, sequence_length=None, beam_width=10, top_paths=1):

    if sequence_length is None:
      sequence_length = [inputs.shape[1]]
      ishape = inputs.shape
      inputs = np.reshape(inputs, (ishape[1], ishape[0], ishape[2]))
    else:
      inputs = np.stack(inputs, axis=1)

    decode = tf.nn.ctc_beam_search_decoder(inputs=inputs,
                                            sequence_length=sequence_length,
                                            beam_width=beam_width,
                                            top_paths=top_paths)
    decode = tf.sparse.to_dense(decode[0][0])
    
    return decode
  return tf.numpy_function(ctc_decoder_, [inputs,
                                          sequence_length,
                                          beam_width,
                                          top_paths], [tf.int64])


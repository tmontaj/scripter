import nltk
import numpy as np
import tensorflow as tf

def object_error_rate(obj1, obj2):
  def object_error_rate_(obj1, obj2):
    
    if len(obj1.shape) == 1:
      obj1 = obj1.numpy().tolist()
      obj2 = obj2.numpy().tolist()
    else:
      obj1 = obj1.numpy()
      obj2 = obj2.numpy()

      obj1 = [i.tolist() for i in obj1]
      obj1 = [i.tolist() for i in obj2]

    distance = nltk.edit_distance(obj1, obj2, transpositions=False)
    len_ = len(obj1)
    return distance/len_
  return tf.py_function(object_error_rate_,(obj1, obj2),(tf.float32))

class Token_ER(tf.keras.metrics.Metric):
  def __init__(self, object_error_rate, decoder, name="Token_ER", **kwargs):
    super(Token_ER, self).__init__(name=name, **kwargs)
    self.error_rate = object_error_rate
    self.total_ler = 0.0
    self.total_wer = 0.0
    self.decoder = decoder

    self.ler = LER(object_error_rate, decoder)
    self.wer = WER(object_error_rate, decoder)
 
  def update_state(self, y_true, y_pred, sample_weight=None):
    sequence_length = tf.cast(tf.math.ceil(y_true[:,0]/2), tf.int32)
    y_pred = self.decoder(y_pred, sequence_length=sequence_length)
    self.wer.update_state(y_true , y_pred)
    self.ler.update_state(y_true , y_pred)
    return (self.wer.result(), self.ler.result())
        
  def result(self):return (self.wer.result(), self.ler.result())
 
  def reset_states(self):
    self.wer.reset_states()
    self.ler.reset_states()

class LER (tf.keras.metrics.Metric):
  def __init__(self, object_error_rate, decoder, name="LER", **kwargs):
    super(LER, self).__init__(name=name, **kwargs)
    self.error_rate = object_error_rate
    self.total = 0.0
    self.decoder = decoder

 
  # @tf.function
  def update_state(self, y_true, y_pred, sample_weight=None):
    total=0.0

    sequence_length = tf.cast(tf.math.ceil(y_true[:,0]/2), tf.int32)
    y_pred = self.decoder(y_pred, sequence_length=sequence_length)
    
    y_true_shape = tf.shape(y_true)[0]
    for i in tf.range(y_true_shape):
      tf.autograph.experimental.set_loop_options(
        shape_invariants=[(total, tf.TensorShape(None))]
      )
      

      len_y_true = y_true[i][2]
      y_true_ = y_true[i][3:len_y_true+3]
      y_pred_ = tf.gather(y_pred, i)
      total = total+self.calc_ler(y_true_, y_pred_)
    self.total = total/tf.cast(y_true_shape, tf.float32)
    
    return self.total
        
  def result(self):return self.total 
 
  @tf.function
  def calc_ler(self, y_true, y_pred):
    return self.error_rate(y_true, y_pred)
 
  def reset_states(self):
    self.total = 0.0


class WER(tf.keras.metrics.Metric):
  def __init__(self, object_error_rate, decoder, space=28, name="WER", **kwargs):
    super(WER, self).__init__(name=name, **kwargs)
    self.error_rate = object_error_rate
    self.total = 0.0
    self.space = space
    self.decoder = decoder


  def update_state(self, y_true, y_pred, sample_weight=None):
    total=0.0
    y_true_shape = tf.shape(y_true)[0]
    # y_pred = tf.gather(y_pred, 0)
    sequence_length = tf.cast(tf.math.ceil(y_true[:,0]/2), tf.int32)
    y_pred = self.decoder(y_pred, sequence_length=sequence_length)

    for i in tf.range(y_true_shape):
      tf.autograph.experimental.set_loop_options(
        shape_invariants=[(total, tf.TensorShape(None))]
      )

      y_pred_ = tf.gather(y_pred, i)
      space   = tf.where(y_pred_==self.space)
      space   = tf.squeeze(space) 

      # 1) add len(x) to space if needed (DONE)
      # 2) make x[n] = x[n]-x[n-1] ()
      # 3) for x[0] = x[1] - 0 
      # 4) split
      def split_(x, space):
        if tf.equal(tf.size(space), 0):
          return x
        space = space.numpy().tolist()
        if space[-1] != x.shape[0]:
          space.append(x.shape[0])
        tmp_space = space[:-1]
        tmp_space.insert(0, 0)
        space = np.array(space)
        tmp_space = np.array(tmp_space)
        space = space-tmp_space

        return tf.split(x, space)

      split = lambda x, space : tf.py_function(split_,(x, space),(tf.int64))
      
      y_pred_ = split(y_pred_, space)

      len_y_true = y_true[i][2]
      y_true_ = y_true[i][3:len_y_true+3]
      space   = tf.where(y_true_==self.space)
      space   = tf.squeeze(space)
      
      y_true_ = split(tf.cast(y_true_, tf.int64), space)
      total+=self.calc_wer(y_true_, y_pred_)

    self.total=total/tf.cast(y_true_shape, tf.float32)

    return self.total

  def result(self):return self.total
  
  # def np2list(self, np_arr): return [i.tolist() for i in np_arr]

  def calc_wer(self, y_true, y_pred):
    # y_true = self.np2list(y_true)
    # y_pred = self.np2list(y_pred)
    return self.error_rate(y_true, y_pred)

  def reset_states(self):
    self.total = 0.0
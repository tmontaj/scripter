import nltk
import numpy as np
import tensorflow as tf

def object_error_rate(obj1, obj2):
  distance = nltk.edit_distance(obj1, obj2, transpositions=False)
  len_ = len(obj1)
  return distance/len_

class Token_ER(tf.keras.metrics.Metric):
  def __init__(self, object_error_rate, decoder, name="Token_ER", **kwargs):
    super(Token_ER, self).__init__(name=name, **kwargs)
    self.error_rate = object_error_rate
    self.total_LER = 0.0
    self.total_WER = 0.0
    self.decoder = decoder

    self.ler = LER(object_error_rate)
    self.wer = WER(object_error_rate)
 
  def update_state(self, y_true, y_pred, sample_weight=None):
    sequence_length = tf.cast(tf.math.ceil(y_true[:,0]/2), tf.int32)
    print("sequence_length", sequence_length)
    print("y_pred", y_pred.shape)
    y_pred = self.decoder(y_pred, sequence_length=sequence_length)
    y_pred = tf.sparse.to_dense(y_pred[0][0])
    print("y_pred",y_pred)
    self.total_wER = self.wer.update_state(y , y_pred)
    self.total_LER = self.ler.update_state(y , y_pred)
    return (self.total_wER, self.total_wER)
        
  def result(self):return (self.total_wER, self.total_wER)
 
  def reset_states(self):
    self.total_wER = 0
    self.total_wER = 0
    self.wer.reset_states()
    self.ler.reset_states()

class LER (tf.keras.metrics.Metric):
  def __init__(self, object_error_rate, name="LER", **kwargs):
    super(LER, self).__init__(name=name, **kwargs)
    self.error_rate = object_error_rate
    self.total = 0.0
 
  def update_state(self, y_true, y_pred, sample_weight=None):
    total=0
    for i in range(y_true.shape[0]):
      len_y_true = y_true[i][0]
      y_true_ = y_true[i][1:len_y_true+1]
      total+=self.calc_ler(y_true_, y_pred[i])
    
    self.total=total/y_true.shape[0]
    return self.total
        
  def result(self):return self.total 
 
  def calc_ler(self, y_true, y_pred):
    return self.error_rate(y_true.numpy(), y_pred.numpy())
 
  def reset_states(self):
    self.total = 0.0


class WER (tf.keras.metrics.Metric):
  def __init__(self, object_error_rate, space= 28, name="WER", **kwargs):
    super(WER, self).__init__(name=name, **kwargs)
    self.error_rate = object_error_rate
    self.total = 0.0
    self.space = space

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    total=0
    for i in range(y_true.shape[0]):
      space = np.argwhere(y_pred[i]==self.space)
      y_pred_ = np.split(y_pred[i], space.squeeze())

      len_y_true = y_true[i][0]
      y_true_ = y_true[i][1:len_y_true+1]
      space = np.argwhere(y_true_==self.space)
      y_true_ = np.split(y_true_, space.squeeze())

      total+=self.calc_wer(y_true_, y_pred_)

    self.total=total/y_true.shape[0]
    return self.total

  def result(self):return self.total
  
  def np2list(self, np_arr): return [i.tolist() for i in np_arr]

  def calc_wer(self, y_true, y_pred):
    y_true = self.np2list(y_true)
    y_pred = self.np2list(y_pred)
    return self.error_rate(y_true, y_pred)

  def reset_states(self):
    self.total = 0.0
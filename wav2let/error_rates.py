import nltk
import numpy as np
import tensorflow as tf

def object_error_rate(obj1, obj2):
  distance = nltk.edit_distance(obj1, obj2, transpositions=False)
  len_ = obj1.shape[0]
  return distance/len_

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
    print(y_true)
    return self.error_rate(y_true, y_pred)
 
  def reset_states(self):
    self.total = 0.0


class WER (tf.keras.metrics.Metric):
  def __init__(self, object_error_rate, space=25, name="WER", **kwargs):
    super(WER, self).__init__(name=name, **kwargs)
    self.error_rate = object_error_rate
    self.total = 0.0
    self.space = space
    # self.vnp2list = np.vectorize(self.np2list)

  def update_state(self, y_true, y_pred, sample_weight=None):
    total=0
    for i in range(y_true.shape[0]):
      space = np.argwhere(y_pred[i]==self.space)
      y_pred_ = np.split(y_pred[i], space.squeeze())
      print("y_pred_",y_pred_)

      len_y_true = y_true[i][0]
      y_true_ = y_true[i][1:len_y_true+1]
      print("b",y_true_.shape)
      # y_true_ = np.delete(y_true_, [0], axis=0)
      print("a",y_true_.shape)
      space = np.argwhere(y_true_==self.space)
      y_true_ = np.split(y_true_, space.squeeze())

      print("y_true_", y_true_)

      total+=self.calc_wer(y_true_, y_pred_)

    self.total=total/y_true.shape[0]
    return self.total

  def result(self):return self.total
  
  def np2list(self, np_arr): return np.array([i.tolist() for i in np_arr])

  def calc_wer(self, y_true, y_pred):
    y_true = self.np2list(y_true)
    y_pred = self.np2list(y_pred)
    return self.error_rate(y_true, y_pred)

  def reset_states(self):
    self.total = 0.0
import tensorflow as tf

def ctc_loss():
  def ctc_loss_(y_true, y_pred):

    label_length = y_true[0]
    true_labels  = y_true[0]

    batch = tf.shape(y_pred)[0] # shape=(batch, time, char)
    char = tf.shape(y_pred)[2] # shape=(batch, time, char)
    logit_length = tf.repeat([char], batch)

    return tf.nn.ctc_loss(labels=true_labels, logits=y_pred,label_length=label_length,
                          logit_length=logit_length)
  return ctc_loss_
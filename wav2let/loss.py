'''
loss for wav2letter model (CTC loss)
'''
import tensorflow as tf

def ctc_loss(REAL_BATCH_SIZE, strategy):
    with strategy.scope():
        def ctc_loss_(y_true, y_pred):
            '''
            CTC loss function
            Arguments:
            y_true -- true labels (text, length of text).
            y_pred -- predected labels list of true labels.
            '''
            label_length = y_true[0]
            true_labels = y_true[0]

            batch = tf.shape(y_pred)[0]  # shape=(batch, time, char)
            char = tf.shape(y_pred)[2]  # shape=(batch, time, char)
            logit_length = tf.repeat([char], batch)

            ctc = tf.nn.ctc_loss(labels=true_labels, logits=y_pred, label_length=label_length,
                                logit_length=logit_length)
            
            return tf.nn.compute_average_loss(ctc, 
                                    global_batch_size=REAL_BATCH_SIZE)
    
    return ctc_loss_

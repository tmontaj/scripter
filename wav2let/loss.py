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
            # tf.print("y_true", tf.shape(y_true))
            # tf.print("y_pred", tf.shape(y_pred))
            label_length = y_true[:,2]
            # tf.print("label_length", tf.shape(label_length))
            true_labels = y_true[:,3:]
            # tf.print("true_labels", tf.shape(true_labels))

            # batch = tf.shape(y_pred)[0]  # shape=(batch, time, char)
            #tf.print("batch",batch)
            # char = tf.shape(y_pred)[2]  # shape=(batch, time, char)
            #tf.print("char",char)
            # logit_length = tf.repeat([char], batch)
            #tf.print("logit_length",tf.shape(logit_length))
            logit_length = y_true # tf.math.ceil(y_true[:,0]/2)

            ctc = tf.nn.ctc_loss(labels=true_labels, logits=y_pred, label_length=label_length,
                                logit_length=logit_length, logits_time_major=False, blank_index=30)
            
            # return tf.nn.compute_average_loss(ctc, 
            #                         global_batch_size=REAL_BATCH_SIZE)
            return ctc
    
    return ctc_loss_

def simple_ctc_loss(y_true, y_pred):
            '''
            CTC loss function
            Arguments:
            y_true -- true labels (text, length of text).
            y_pred -- predected labels list of true labels.
            '''
            label_length = y_true[:,2]
            true_labels = y_true[:,3:]

            logit_length = y_true[:, 0] 

            ctc = tf.nn.ctc_loss(labels=true_labels, logits=y_pred, label_length=label_length,
                                logit_length=logit_length, logits_time_major=False, blank_index=30)
            
            return ctc
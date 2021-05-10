
import os
import tensorflow as tf
from wav2let.error_rates_non_graph import *
from wav2let.ctc_decoder import ctc_decoder


def get_last_model(path):
    models = os.listdir(path)
    last = 0
    for i in range(1, len(models)):
        model = models[i].split("_")
        model = model[1]
        model = model.split(".")
        if last < int(model[0]):
            last = int(model[0])

    return last

def load_model(path):
    last = get_last_model(path)
    model = tf.keras.models.load_model("path"+"/"+"_"+str(last)+".h5")
    return model

def calculate_WER(model, object_error_rate=object_error_rate, data):
    wer = WER(object_error_rate)
    total_wer = 0
    real_wer = total_wer/i
    i = 1
    for x,y in data:
        y_pred = model.predict(x)
        wer.update_state(y , y_pred)
        total_wer += wer.result()
        print("WER per patch",i, "  ", wer.result())
    
    return real_wer

def calculate_LER(model, object_error_rate=object_error_rate, data):
    ler = LER(object_error_rate)
    total_ler = 0
    real_ler = total_ler/i
    i = 1
    for x,y in data:
        y_pred = model.predict(x)
        ler.update_state(y , y_pred)
        total_ler += ler.result()
        print("LER per patch",i, "  ", ler.result())
    
    return real_ler

def predict_sample(model, decoder=ctc_decoder, sample):
    y_pred = model.predict(sample)

    y_true_shape = tf.shape(y)[0]
    # y_pred = tf.gather(y_pred, 0)
    sequence_length = tf.cast(tf.math.ceil(y[:,0]/2), tf.int32)
    y_pred = self.decoder(y_pred, sequence_length=sequence_length)

        
def one_hot_decode(one_hot):
    index = tf.argmax(one_hot, axis=0)

    return index

def int2string(ints, alphabet_size = 26, first_letter=97, len_=True):
   
    ints = ints[ints != 0]
    ints = ints+first_letter-1

    ints = tf.where(ints==(first_letter-1)+alphabet_size+1, 46, ints) # replace dot
    ints = tf.where(ints==(first_letter-1)+alphabet_size+2, 32, ints) # replace space
    ints = tf.where(ints==(first_letter-1)+alphabet_size+3, 44, ints) # replace comma

    ints = tf.strings.unicode_encode(ints, output_encoding="UTF-8")
    # ints = ints.numpy().decode('UTF-8')
    return ints

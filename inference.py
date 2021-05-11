
import os
import tensorflow as tf
from usecases.hprams.test import data_hprams  # pylint: disable=imports
from wav2let.error_rates_non_graph import *
from wav2let.ctc_decoder import ctc_decoder as decoder
from wav2let.model import Wav2Let  # pylint: disable=imports
from wav2let.loss import ctc_loss as loss  # pylint: disable=imports
import dataset.pipeline.librispeech as pipeline
from dataset.data.load.librispeech import load  # pylint: disable=imports
from dataset.data.load.libri_what_to_download import what_to_download as wtd  # pylint: disable=imports
from dataset.data.load import safe_load  # pylint: disable=imports
import pickle

def get_last_model(path):
    models = os.listdir(path)
    models = sorted([int(i[1:]) for i in models if i != "temp"])    
    return models[-1]

def load_last_model(path):
    name = get_last_model(path)
    return load_model(path, name)

def load_best_model(path):
    return load_model(path, name="best")

def init_model():
    # note you have to change fake sample in case
    # you have changed the MFCC parametars
    model = Wav2Let()
    fake_sample = pickle.load(open("sample.io", "rb"))
    model(fake_sample[0])

    return model

def load_model(path, name):
    model = init_model()
    model.load_weights(path+"/"+"_"+str(name)+".h5")
    return model

def int2string(ints, alphabet_size = 26, first_letter=97, len_=True):
   
    ints = ints[ints != 0]
    ints = ints+first_letter-1

    ints = tf.where(ints==(first_letter-1)+alphabet_size+1, 46, ints) # replace dot
    ints = tf.where(ints==(first_letter-1)+alphabet_size+2, 32, ints) # replace space
    ints = tf.where(ints==(first_letter-1)+alphabet_size+3, 44, ints) # replace comma

    ints = tf.strings.unicode_encode(ints, output_encoding="UTF-8")
    ints = ints.numpy().decode('UTF-8')
    return ints


def predict_sample(model, sample):
    if len(sample.shape)==2:
        sample = tf.expand_dims(sample, 0)
    
    y_pred = model(sample)
    y_pred = decoder(y_pred, sequence_length=[y_pred.shape[1]])
    y_pred = tf.cast(y_pred[0], tf.int32)
    y_pred = int2string(y_pred)

    return y_pred

# model = load_model("../weights/wav2letter", 0)
# sample = pickle.load(open("sample.io", "rb"))[0]
# y_pred_0 = predict_sample(model, sample[0])
# y_pred_1 = predict_sample(model, sample[1])

# print("y_pred_0")
# print(y_pred_0)
# print("y_pred_1")
# print(y_pred_1)

def one_hot_decode(one_hot):
    index = tf.argmax(one_hot, axis=0)
    return index



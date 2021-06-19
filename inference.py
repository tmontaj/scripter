
import os
import tensorflow as tf
from usecases.hprams.test import data_hprams  # pylint: disable=imports
from wav2let.error_rates_non_graph import *
from wav2let.ctc_decoder import ctc_decoder as decoder
from wav2let.model import Wav2Let  # pylint: disable=imports
from wav2let.loss import ctc_loss as loss  # pylint: disable=imports
import pickle
import urllib.parse


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


def predict_sample(model, sample, len_):

    if len(sample.shape)==2:
        sample = tf.expand_dims(sample, 0)
    
    sample = sample[:len_]
    y_pred = model(sample)
    
    y_pred = decoder(y_pred, sequence_length=[y_pred.shape[1]])
    y_pred = tf.cast(y_pred[0], tf.int32)
    y_pred = int2string(y_pred)

    return y_pred

def clone_dataset():
    '''
    Clone the dataset repo to ../dataset

    Arguments:
    username_ -- github username
    password_ -- github password
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))

    folder_name = "dataset"
    git = "git clone https://github.com/tmontaj/Text-AudioDatasets.git" 
    path = " "+dir_path + "/../" + folder_name
    comand = git + path
    os.system(comand)

    pip = "pip install -r dataset/requirements.txt"
    os.system(pip)


def data_pipline(idx):
    '''
    Download the dataset splits to ../dataset/dataset

    Returns:
    data -- split data
    '''
    # pass # use test code here
    home = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(home, "dataset")  # dataset repo link
    os.system("mkdir -p %s/dataset/librispeech" % (src))
    src = os.path.join(src, "dataset")  # dataset actual recordes link
    safe_load(load, wtd, src, ["dev-clean"]) # train-clean-100
    batch = data_hprams["batch"] 
    
    # train = pipeline.text_audio(
    #     src=src, split="dev-clean", batch=batch, **data_hprams["audio2text"])

    train = pipeline.load_sample_text_audio(
        src=src, idx=idx, split="dev-clean", batch=batch, **data_hprams["audio2text"])

    
    return train


if __name__ == '__main__':

    clone_dataset()

    from dataset.data.load.librispeech import load  # pylint: disable=imports
    from dataset.data.load.libri_what_to_download import what_to_download as wtd  # pylint: disable=imports
    from dataset.data.load import safe_load  # pylint: disable=imports
    import dataset.pipeline.librispeech as pipeline  # pylint: disable=imports 
    from dataset.data.transform.text import int2string  # pylint: disable=imports 

    sample, text = data_pipline(0)
    
    model = load_model("../weights/wav2letter", 80)
    len_ = text[0]
    text = text[3:]
    y_pred_0 = predict_sample(model, sample, len_)

    print(int2string(text[3:]))
    print("y_pred_0")
    print(y_pred_0)

    # print(int2string(text))
    # print(text)

    # sample = pickle.load(open("sample.io", "rb"))[0]
    # y_pred_1 = predict_sample(model, sample[1])

    # print("y_pred_1")
    # print(y_pred_1)



def one_hot_decode(one_hot):
    index = tf.argmax(one_hot, axis=0)
    return index



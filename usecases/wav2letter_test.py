'''
Test file for wav2letter uses dev-clean split of
librispeach data set as it is small.

This split is used as both train and val sets.
The the train loop is used

The aim of this file is to run tests only NEVER use it
for real training
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hprams.test import data_hprams  # pylint: disable=imports
from hprams.test import hcallbacks  # pylint: disable=imports
from wav2let.model import Wav2Let  # pylint: disable=imports
from wav2let.loss import ctc_loss  # pylint: disable=imports
from wav2let.fit import fit  # pylint: disable=imports
import urllib.parse
import tensorflow as tf


def clone_dataset(username_, password_):
    '''
    Clone the dataset repo to ../dataset

    Arguments:
    username_ -- github username
    password_ -- github password
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))

    username_ = urllib.parse.quote(username_)
    folder_name = "dataset"
    git = "git clone https://%s:%s@github.com/tmontaj/Text-AudioDatasets.git" % (
        username_, password_)
    path = " "+dir_path + "/../" + folder_name
    comand = git + path
    os.system(comand)

    pip = "pip install -r dataset/requirements.txt"
    os.system(pip)


def data_pipline(strategy):
    '''
    Download the dataset splits to ../dataset/dataset

    Returns:
    data -- split data
    '''
    # pass # use test code here
    home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(home, "dataset")  # dataset repo link
    os.system("mkdir -p %s/dataset/librispeech" % (src))
    src = os.path.join(src, "dataset")  # dataset actual recordes link
    safe_load(load, wtd, src, ["dev-clean"]) # train-clean-100
    BATCH_SIZE_PER_REPLICA = data_hprams["batch"]
    REAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    train = pipeline.text_audio(
        src=src, split="dev-clean", batch=REAL_BATCH_SIZE, **data_hprams["audio2text"])
    dev   = pipeline.text_audio(
        src=src, split="dev-clean", batch=REAL_BATCH_SIZE, **data_hprams["audio2text"])
    return train, dev, REAL_BATCH_SIZE
  

def train_test(username, password):
    '''
    Train loop (save metric ...etc to W&B)
    '''
    strategy = tf.distribute.MirroredStrategy()
    print("*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_")
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    train_set, dev, gbs = data_pipline(strategy)
    # data = data.take(3)
    # print("data", data)
    # for i in data:
    #     print("sample", i)
    print("gbs")
    print(gbs)
    train_set = strategy.experimental_distribute_dataset(train_set)
    dev = strategy.experimental_distribute_dataset(dev)

    n_epochs = 500000
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(dir_path, "..", "..",
                             "weights", "wav2letter")
    with strategy.scope():
        optimizer = tf.optimizers.Adam()
        model = Wav2Let()
    loss = ctc_loss(REAL_BATCH_SIZE=gbs, strategy=strategy)
    hcallbacks["lr"]=0.001
    hcallbacks["username_"]=username
    hcallbacks["password_"]=password
    fit(train_set=train_set, val_set=dev, n_epochs=n_epochs, model=model,
        optimizer=optimizer, loss=loss, save_path=save_path,
        strategy=strategy, hcallbacks=hcallbacks, restart=True)


if __name__ == '__main__':

    username = sys.argv[1:][0]
    password = sys.argv[1:][1]

    clone_dataset(username, password)

    from dataset.data.load.librispeech import load  # pylint: disable=imports
    from dataset.data.load.libri_what_to_download import what_to_download as wtd  # pylint: disable=imports
    from dataset.data.load import safe_load  # pylint: disable=imports
    import dataset.pipeline.librispeech as pipeline  # pylint: disable=imports

    train_test(username, password)

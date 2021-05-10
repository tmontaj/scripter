import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hprams.test import data_hprams  # pylint: disable=imports
from hprams.test import hcallbacks  # pylint: disable=imports
from wav2let.model import Wav2Let  # pylint: disable=imports
from wav2let.loss import ctc_loss  # pylint: disable=imports
from wav2let.error_rates import *
from wav2let.ctc_decoder import ctc_decoder
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


def data_pipline_dev(strategy):
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
    safe_load(load, wtd, src, ["dev-clean"])
    BATCH_SIZE_PER_REPLICA = data_hprams["batch"]
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    REAL_BATCH_SIZE = GLOBAL_BATCH_SIZE * data_hprams["batch"]
    data = pipeline.text_audio(
        src=src, split="dev-clean", batch=GLOBAL_BATCH_SIZE, **data_hprams["audio2text"])
    return data, REAL_BATCH_SIZE

def data_pipline_train(strategy):
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
    safe_load(load, wtd, src, ["dev-other"])
    BATCH_SIZE_PER_REPLICA = data_hprams["batch"]
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    REAL_BATCH_SIZE = GLOBAL_BATCH_SIZE * data_hprams["batch"]
    data = pipeline.text_audio(
        src=src, split="dev-other", batch=GLOBAL_BATCH_SIZE, **data_hprams["audio2text"])
    return data, REAL_BATCH_SIZE


def test():
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    dev_data, gbs = data_pipline_dev(strategy)
    dev_data = dev_data.take(1)
    #train_data, gbs = data_pipline_train(strategy)

    model = Wav2Let()
    loss = ctc_loss(REAL_BATCH_SIZE=gbs, strategy=strategy)
    optimizer = tf.optimizers.Adam()

    model.compile(loss = loss, optimizer=optimizer)

    ter = WER(object_error_rate=object_error_rate, decoder=ctc_decoder)
    for x,y in dev_data:
        y_pred = model.predict(x)
        ter.update_state(y , y_pred)
        print("ter",ter.result())
        



if __name__ == '__main__':

    username = sys.argv[1:][0]
    password = sys.argv[1:][1]

    clone_dataset(username, password)

    from dataset.data.load.librispeech import load  # pylint: disable=imports
    from dataset.data.load.libri_what_to_download import what_to_download as wtd  # pylint: disable=imports
    from dataset.data.load import safe_load  # pylint: disable=imports
    import dataset.pipeline.librispeech as pipeline  # pylint: disable=imports

    test()


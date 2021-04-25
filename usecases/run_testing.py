import os
import sys
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
    safe_load(load, wtd, src, ["dev-clean"])
    BATCH_SIZE_PER_REPLICA = data_hprams["batch"]
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    REAL_BATCH_SIZE = GLOBAL_BATCH_SIZE * data_hprams["batch"]
    data = pipeline.text_audio(
        src=src, split="dev-clean", batch=GLOBAL_BATCH_SIZE, **data_hprams["audio2text"])
    return data, REAL_BATCH_SIZE

def test():
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    data, gbs = data_pipline(strategy)
    sample = data.take(1)

    path = "/home/ubuntu/payload/weights/wav2letter"
    model = load_model(path)
    
    WER = calculate_WER(model= model, data= data)
    print("WER for dev data", WER)
    LER = calculate_LER(model= model, data= data)
    print("LER for dev data", LER)

    

        



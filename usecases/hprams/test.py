'''hprams for testing'''
data_hprams = {
    "audio2text": {
        "reverse": "False",
        "len_": True,
        "batch": 2,
        "threshold": 5,
        "is_spectrogram": True,
        "remove_comma": False,
        "alphabet_size": 26,
        "first_letter": 96,
        "melspectrogram": {
            "nfft": 800,
            "window": 512,
            "stride": 200,
            "mels": 80,
            "fmin": 0,
            "fmax": 8000
        }
    },

    "batch": 2,
}

hcallbacks = {
    "n_epoch": 3,
    "patience": 3,
    "factor": 0.1,
    "patience_plateau": 4,
}

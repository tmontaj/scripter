'''hprams for testing'''
data_hprams = {
    "audio2text": {
        "reverse": True,
        "len_": True,
        "threshold": 5,
        "is_spectrogram": True,
        "remove_comma": False,
        "alphabet_size": 26,
        "first_letter": 97,
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
    "n_epoch": 10,
    "patience": 3,
    "factor": 0.1,
    "patience_plateau": 4,
    "file_path":"./wav2let/manual_hprams_tune/var.json",
}

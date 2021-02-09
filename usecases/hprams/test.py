'''hprams for testing'''
data_hprams = {
    "speaker_verification": {
        "threshold": 5,
        "sampling_rate": 16000,
        "buffer_size": 1000,
        "num_recordes": 3,
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

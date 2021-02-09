# io_channels = 128
# hidden_channels = 128

# sample_rate= 24000 # input will be standardised
#  to this rate

# fft_step   = 12.5/1000. # 12.5ms
# fft_window = 50.0/1000.  # 50ms

# n_fft = 512*4

# hop_length = int(fft_step*sample_rate)
# win_length = int(fft_window*sample_rate)

# n_mels = 80
# fmin = 125 # Hz
# #fmax = ~8000

# #np.exp(-7.0), np.log(spectra_abs_min)  # "Audio tests"
# suggest a min log of -4.605 (-6 confirmed fine)
# spectra_abs_min = 0.01

# mel_bins, spectra_bins = n_mels, n_fft//2+1 # 80, 1025
# #steps_total, steps_leadin = 1024, 64
# steps_total, steps_leadin = 549, 64

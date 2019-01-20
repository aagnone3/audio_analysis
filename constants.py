import numpy as np
from numpy.fft import fftfreq, fft, fftshift

FRAME_SIZE = 1024
N_CHANNELS = 1
N_FRAME_DISPLAY = 200
WINDOW = np.hamming(FRAME_SIZE)
FS = 44100
FFT_FREQS = FS * fftshift(fftfreq(FRAME_SIZE))

import pyaudio
import time
import numpy as np
from matplotlib import pyplot as plt

from constants import (
    N_CHANNELS,
    FS,
    FRAME_SIZE
)


class Microphone():

    def __init__(self, signal):
        self.signal = signal
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                            channels=N_CHANNELS,
                            rate=FS,
                            input=True,
                            frames_per_buffer=FRAME_SIZE)

    def sample(self):
        data = self.stream.read(FRAME_SIZE)
        y = np.fromstring(data, np.float32)
        self.signal.emit(y)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

import pyaudio
import time
import numpy as np
from matplotlib import pyplot as plt

from sigproc import get_frame_size


class Microphone():

    defaults = {
        "n_channels": 1,
        "fs": 44100,
        "frame_size_ms": 50
    }

    def __init__(self, signal, **kwargs):
        # default parameters if not specified
        for name, val in self.defaults.items():
            setattr(self, name, kwargs.get(name, val))

        self.signal = signal
        self.p = pyaudio.PyAudio()
        self.frame_size = get_frame_size(self.fs, self.frame_size_ms)

        self.stream = self.p.open(format=pyaudio.paFloat32,
                            channels=self.n_channels,
                            rate=self.fs,
                            input=True,
                            frames_per_buffer=self.frame_size)

    def sample(self):
        data = self.stream.read(self.frame_size)
        y = np.fromstring(data, np.float32)
        self.signal.emit(y)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

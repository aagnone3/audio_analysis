import numpy as np
from numpy.fft import rfftfreq, fft, fftshift, rfft
import pyqtgraph as pg
from matplotlib import cm
from pyqtgraph.Qt import QtCore

from sigproc import get_frame_size


class SpectralFrameWidget(pg.PlotWidget):

    defaults = {
        "fs": 44100,
        "nfft": 44100,
        "frame_size_ms": 50
    }

    def __init__(self, **kwargs):
        super(SpectralFrameWidget, self).__init__()

        # default parameters if not specified
        for name, val in self.defaults.items():
            setattr(self, name, kwargs.get(name, val))

        self.setLabel('left', 'Amplitude', units='dB')
        self.setLabel('bottom', text='Frequency', units='kHz')
        self.getPlotItem().setRange(yRange=[-50, 40])
        self.spectrum_curve = self.plot([])
        self.freqs = rfftfreq(self.nfft) * self.fs / 1000.0
        self.frame_size = get_frame_size(self.fs, self.frame_size_ms)
        self.window = np.hamming(self.frame_size)

    def update(self, chunk):
        spectral_frame = np.abs(rfft(self.window * chunk, n=self.nfft))
        spectral_frame = 20 * np.log10(spectral_frame + 1e-8)
        self.spectrum_curve.setData(self.freqs, spectral_frame)



# Based off of implementation from github user boylea
# https://gist.github.com/boylea/1a0b5442171f9afbf372
class SpectrogramWidget(pg.PlotWidget):

    defaults = {
        "display_width_seconds": 20,
        "fs": 44100,
        "nfft": 2048,
        "frame_size_ms": 50
    }

    def __init__(self, **kwargs):
        super(SpectrogramWidget, self).__init__()

        # default parameters if not specified
        for name, val in self.defaults.items():
            setattr(self, name, kwargs.get(name, val))

        # default f_max based on fs if not provided.
        # if it is provided, validate it.
        f_max = kwargs.get("f_max_display")
        if f_max is None:
            f_max = self.fs // 2
        elif f_max > self.fs // 2:
            raise ValueError("f_max must be <= fs // 2")
        self.f_max = f_max

        self.n_freq_display = int(self.nfft * self.f_max / float(self.fs // 2))
        self.n_freq_display = int(self.n_freq_display/2+1)

        self.window_size = get_frame_size(self.fs, self.frame_size_ms)
        self.n_frame_display = int(self.display_width_seconds * self.fs / float(self.nfft))
        print("# frames displayed: {}".format(self.n_frame_display))
        freqs = np.arange(self.n_freq_display) * self.fs / float(self.nfft)
        print("Displaying {} values of a {}-pt FFT: freqs [{:.1f},{:.1f}]".format(
            self.n_freq_display*2, self.nfft, freqs.min(), freqs.max()))
        self.plot_roll_amount = self.n_freq_display
        self.spec_shape = (self.n_frame_display, self.n_freq_display)
        print("Spectrogram shape: {}".format(self.spec_shape))

        self.img = pg.ImageItem()
        self.addItem(self.img)

        # TODO AGC
        self.psd_levels = [-10, 40]
        self.img_array = np.ones(self.spec_shape) * self.psd_levels[0]
        self.img.setLookupTable(self._get_color_lut())
        self.img.setLevels(self.psd_levels)

        # setup the correct scaling for y-axis
        yscale = 1.0 / (self.img_array.shape[1] / freqs[-1])
        self.img.scale(self.n_freq_display * 2 / float(self.fs), yscale)
        self.setLabel('left', 'Frequency', units='Hz')
        self.setLabel('bottom', text='Time', units='s')

        # prepare short-time window for later use
        self.window = np.hamming(self.window_size)
        self.show()

    def _get_color_lut(self):
        # colormap = cm.get_cmap("nipy_spectral")
        colormap = cm.get_cmap("gray")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        return lut

    def update(self, chunk):
        # normalized, windowed frequencies in data chunk
        spec = np.fft.rfft(chunk * self.window, n=self.nfft)
        spec = spec[:self.n_freq_display]

        # convert to dB scale
        psd = 20.0 * np.log10(np.abs(spec) + 1e-8)

        # roll down and replace leading indices with new data
        self.img_array = np.roll(self.img_array, -1, axis=0)
        self.img_array[-1, :] = psd

        # if np.random.normal() < -2:
        #     print(self.img_array.min(), self.img_array.max())
        #     print(np.percentile(psd, 95))

        self.img.setImage(self.img_array, autoLevels=False)

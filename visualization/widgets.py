import numpy as np
from numpy.fft import fftfreq, fft, fftshift
import pyqtgraph as pg
from matplotlib import cm
from pyqtgraph.Qt import QtCore

from constants import (
    WINDOW,
    FFT_FREQS
)


class SpectralFrameWidget(pg.PlotWidget):

    def __init__(self):
        super(SpectralFrameWidget, self).__init__()
        self.setLabel('bottom', text='Frequency')
        self.getPlotItem().setRange(yRange=[-100, 10])
        self.spectrum_curve = self.plot([])

    def update(self, chunk):
        spectral_frame = np.abs(fftshift(fft(WINDOW * chunk)))
        spectral_frame[spectral_frame == 0] = 10 ** -10
        spectral_frame = 20 * np.log10(spectral_frame)
        # TODO one-sided
        self.spectrum_curve.setData(FFT_FREQS, spectral_frame)



# Based off of implementation from github user boylea
# https://gist.github.com/boylea/1a0b5442171f9afbf372
class SpectrogramWidget(pg.PlotWidget):

    def __init__(self, window_size, n_frame_display, fs):
        super(SpectrogramWidget, self).__init__()

        self.window_size = window_size
        self.fs = fs
        self.n_frame_display = n_frame_display
        self.plot_roll_amount = int(window_size / n_frame_display)

        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.img_array = np.zeros((window_size, int(self.window_size/2+1)))
        self.img.setLookupTable(self._get_color_lut())
        self.img.setLevels([-1000, 0])

        # setup the correct scaling for y-axis
        freq = np.arange((self.window_size/2)+1)/(float(self.window_size)/self.fs)
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        self.img.scale((1./self.fs)*self.window_size, yscale)
        self.setLabel('left', 'Frequency', units='Hz')

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
        spec = np.fft.rfft(chunk * self.window) / self.window_size

        # convert to dB scale
        spec[spec == 0] = 10**-10
        psd = np.log(np.abs(spec))

        # roll down and replace leading indices with new data
        self.img_array = np.roll(self.img_array, -self.plot_roll_amount, 0)
        self.img_array[-self.plot_roll_amount:] = psd

        # print(self.img_array.min(), self.img_array.max())
        self.img.setImage(self.img_array, autoLevels=False)

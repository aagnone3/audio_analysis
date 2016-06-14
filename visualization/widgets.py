import numpy as np
import pyqtgraph as pg


# Implementation of github user boylea
# https://gist.github.com/boylea/1a0b5442171f9afbf372
class SpectrogramWidget(pg.PlotWidget):

    def __init__(self, window_size, fs):
        super(SpectrogramWidget, self).__init__()

        self.window_size = window_size
        self.fs = fs
        self.plot_roll_amount = 5
        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.img_array = np.zeros((1000, self.window_size/2+1))

        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([
            [0, 255, 255, 255],
            [255, 255, 0, 255],
            [0, 0, 0, 255],
            (0, 0, 255, 255),
            (255, 0, 0, 255)
        ], dtype=np.ubyte)
        color_map = pg.ColorMap(pos, color)
        lut = color_map.getLookupTable(0.0, 1.0, 256)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-50, 40])

        # setup the correct scaling for y-axis
        freq = np.arange((self.window_size/2)+1)/(float(self.window_size)/self.fs)
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        self.img.scale((1./self.fs)*self.window_size, yscale)

        self.setLabel('left', 'Frequency', units='Hz')

        # prepare window for later use
        self.window = np.hamming(self.window_size)
        self.show()

    def update(self, chunk):
        # normalized, windowed frequencies in data chunk
        spec = 1/self.window_size * np.fft.rfft(chunk * self.window)
        # convert to dB scale
        spec[spec == 0] = 10**-10
        psd = 20 * np.log10(abs(spec))

        # roll down and replace leading indices with new data
        self.img_array = np.roll(self.img_array, -self.plot_roll_amount, 0)
        self.img_array[-self.plot_roll_amount:] = psd

        self.img.setImage(self.img_array, autoLevels=False)

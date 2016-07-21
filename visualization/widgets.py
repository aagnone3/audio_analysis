import numpy as np
import pyqtgraph as pg


# Based off of implementation from github user boylea
# https://gist.github.com/boylea/1a0b5442171f9afbf372
class SpectrogramWidget(pg.PlotWidget):

    def __init__(self, window_size, n_frame_display, fs):
        super(SpectrogramWidget, self).__init__()

        self.window_size = window_size
        self.fs = fs
        self.n_frame_display = n_frame_display
        self.plot_roll_amount = window_size / n_frame_display
        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.img_array = np.zeros((window_size, self.window_size/2+1))

        # bipolar colormap
        red = (255, 0, 0, 255)
        green = (0, 255, 0, 255)
        blue = (0, 0, 255, 255)
        yellow = (0, 255, 255, 255)
        black = (0, 0, 0, 0)
        white = (255, 255, 255, 255)
        pos = np.array([0., 0.25, 0.5, 0.75, 1.])
        color = np.array([
            black,
            green,
            blue,
            red,
            white,
        ], dtype=np.ubyte)
        color_map = pg.ColorMap((0.0, 1.0), (black, white))
        lut = color_map.getLookupTable(0.0, 1.0, 256)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-100, 0])

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
        spec = np.fft.rfft(chunk * self.window) / self.window_size

        # convert to dB scale
        spec[spec == 0] = 10**-10
        psd = 20 * np.log10(abs(spec))

        # roll down and replace leading indices with new data
        self.img_array = np.roll(self.img_array, -self.plot_roll_amount, 0)
        self.img_array[-self.plot_roll_amount:] = psd

        self.img.setImage(self.img_array, autoLevels=False)

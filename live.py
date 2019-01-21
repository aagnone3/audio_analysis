import os
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtCore import QObject

from visualization.widgets import (
    SpectrogramWidget,
    SpectralFrameWidget
)
from samplers import Microphone
from constants import (
    FRAME_SIZE,
    N_FRAME_DISPLAY,
    WINDOW,
    FS,
    FFT_FREQS
)


class Observer(QObject):
    """
    Simple class to hold a Qt signal and connect it to multiple observers/consumers.
    """
    signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, *args, **kwargs):
        super(Observer, self).__init__(*args, **kwargs)


def init_gui():
    pg.setConfigOptions(antialias=True)
    app = QtGui.QApplication([])
    app = app.instance()

    win = pg.QtGui.QMainWindow()
    dims = QDesktopWidget().availableGeometry(win)
    win.setWindowTitle('Audio Visualization')
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QVBoxLayout()
    cw.setLayout(layout)
    win.resize(int(0.7 * dims.width()), int(0.7 * dims.height()))
    return app, win, layout


def main():
    app, win, layout = init_gui()

    # initialize plotting widgets
    spectral_frame_widget = SpectralFrameWidget()
    spectrogram = SpectrogramWidget(FRAME_SIZE, N_FRAME_DISPLAY, FS)

    # set up data inter-communication signals
    observer = Observer()
    audio_update = observer.signal
    mic = Microphone(audio_update)
    audio_update.connect(spectrogram.update)
    audio_update.connect(spectral_frame_widget.update)

    # add all widgets to the layout
    layout.addWidget(spectral_frame_widget)
    layout.addWidget(spectrogram)

    # start sampling
    interval = FS / FRAME_SIZE
    ms_interval = 1000.0 / interval
    print("Sampling at sr={}, audio fps={:.3f}, timer interval={:.1f}ms".format(FS, interval, ms_interval))
    t = QtCore.QTimer()
    t.timeout.connect(mic.sample)
    t.start(1000 / interval)

    win.show()
    app.exec_()
    mic.close()


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
import wave

import numpy as np
import pyaudio
import pyqtgraph as pg
from numpy.fft import fftfreq, fft, fftshift
from pyqtgraph.Qt import QtGui

from common.audio import read_audio_file
from visualization.widgets import SpectrogramWidget

# constants
WINDOW_SIZE = 1024
WINDOW = np.hamming(WINDOW_SIZE)

# open files and read initial data
ind = 0
path_to_file = "res/audio/apollo.wav"
p = pyaudio.PyAudio()
wf = wave.open(path_to_file, 'rb')
[fs, wave_values] = read_audio_file(path_to_file)
wave_values = np.array(wave_values)
fft_frequencies = fs * fftshift(fftfreq(WINDOW_SIZE))

# set up plot(s)
app = QtGui.QApplication([])
win = pg.QtGui.QMainWindow()
win.setWindowTitle('Audio Visualization')
cw = QtGui.QWidget()
win.setCentralWidget(cw)
layout = QtGui.QVBoxLayout()
cw.setLayout(layout)

# initialize frequency domain plot
spectrum_plot_widget = pg.PlotWidget()
layout.addWidget(spectrum_plot_widget)
# spectrum_plot.hideAxis('left')
spectrum_plot_widget.setLabel('bottom', text='Frequency')
# spectrum_plot_widget.getPlotItem().setRange(yRange=[0, 50 * wave_values.max()])
spectrum_curve = spectrum_plot_widget.plot([])
# initialize time domain plot
time_plot_widget = pg.PlotWidget()
layout.addWidget(time_plot_widget)
time_curve = time_plot_widget.plot([])
# initialize spectrogram plot
spectrogram = SpectrogramWidget(WINDOW_SIZE, fs)
layout.addWidget(spectrogram)

win.show()


def callback(in_data, frame_count, time_info, status):
    """
    callback function the PyAudio stream calls to retrieve more audio to play
    :param in_data:
    :param frame_count:
    :param time_info:
    :param status:
    :return:
    """
    global spectrum_curve, time_curve, wf, wave_values, fs, fft_frequencies, spectrogram

    # update audio segment to output
    new_sound = wf.readframes(frame_count)
    # values = np.fromstring(new_sound, dtype=np.int32)

    # update fft magnitude
    pos = wf.tell()
    # TODO fix exception on last frame -- shape of fft is not WINDOW_SIZE
    # TODO replace with LP or MFC cepstrum
    fft_frame = abs(fftshift(fft(WINDOW * wave_values[pos:pos + WINDOW_SIZE])))
    fft_frame[fft_frame == 0] = 10 ** -10
    spectrum_curve.setData(fft_frequencies, fft_frame)
    # spectrum_curve.setData(fft_frequencies, 20*np.log10(fft_frame))

    # update time domain signal
    start = max(0, pos - 100*frame_count)
    end = pos
    time_curve.setData(wave_values[start:end])

    # update spectrogram
    spectrogram.update(wave_values[pos:pos + WINDOW_SIZE])

    # play more sound
    return new_sound, pyaudio.paContinue


# open the stream and use the callback function to get more data
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                frames_per_buffer=WINDOW_SIZE,
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                stream_callback=callback)
stream.start_stream()

# keep the thread alive while the stream is still active
while stream.is_active():
    QtGui.QApplication.processEvents()

# stop and close the stream
stream.stop_stream()
stream.close()
# close the wave file
wf.close()
# close PyAudio
p.terminate()

import wave

import numpy as np
import pyaudio
import scipy.io.wavfile as wav_file

from . import environment

# Constants
EPSILON = 0.00000001


def play_file(path_to_file, frame_size=1024):
    """
    Plays a WAV file through the standard output stream.
    :param path_to_file: path to the audio file
    :param frame_size: size of audio frame to output to each data request from the stream
    :return:
    """
    # open file and read initial data
    wf = wave.open(path_to_file, 'rb')
    data = wf.readframes(frame_size)

    # open output audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # play sound through the opened stream
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(frame_size)

    # stop and close stream
    stream.stop_stream()
    stream.close()
    p.terminate()


def read_audio_file(path, mono=True):
    """
    Reads in an audio file and returns a numpy array of the data.
    :param path: path to the audio file
    :param mono: whether to convert the audio to stereo
    :return: audio data and its sampling rate
    """
    # Currently only supports .wav files
    extension = environment.file_extension(path).lower()
    if extension != '.wav':
        raise NotImplementedError("read_audio_file() currently only supports .wav files, not %s" % extension)

    [fs, x] = wav_file.read(path)
    if mono:
        x = stereo2mono(x)

    return fs, x


def frame_split(signal, fs=None, ms_size=None, ms_shift=None):
    """
    Splits the fs-sampled signal into frames of specified size and shift.
    Note that this method does not apply any windowing function to the frames.
    :param signal: signal data, or path to signal
    :param ms_size: duration (ms) of each frame
    :param ms_shift: duration (ms) of each shift between successive frames
    :param fs: sampling frequency used to obtain signal
    :return: numpy array of frames of the signal
    """
    if isinstance(signal, str):
        [fs, x] = read_audio_file(signal)
    else:
        if fs is None:
            raise AttributeError("Must specify fs if passing pre-loaded audio data to frame_split()")
        x = signal

    # convert time-duration to sample-duration
    frame_size = ms_size * fs
    frame_shift = ms_shift * fs
    n_frames = int(len(signal) / frame_shift)
    frames = np.empty((n_frames, frame_size))

    # frame the signal
    i = 0
    signal_pos = 0
    while i < n_frames:
        frames[i] = x[signal_pos:signal_pos + frame_size]
        signal_pos += frame_shift
        i += 1
    return frames, fs


def stereo2mono(x):
    """
    Asserts that the audio sequence is a single channel.
    :param x: audio sequence
    :return: single channel of the data
    """

    if x.ndim == 2:
        return (x[:, 1] / 2) + (x[:, 0] / 2)
    elif x.ndim > 2:
        raise Exception('More than 2 channels passed to stereo2mono() function!')
    else:
        return x

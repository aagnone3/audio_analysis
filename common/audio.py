from __future__ import print_function
import wave
# ensure essentia imports occur before numpy imports. for reasons un-investigated, importing numpy and then
# anything from essentia causes a seg fault. This is likely an essentia bug.
from essentia.standard import Resample, MonoLoader, MonoWriter
import numpy as np
import pyaudio
import environment

# standard representation of an sufficiently small, yet non-zero deviation
EPSILON = 0.00000001


def resample_audio_file(path, desired_fs):
    """
    Re-samples the audio file to the desired sampling rate.
    This overwrites the existing file.
    :param path: path to the existing file.
    :param desired_fs: desired sampling rate for the new audio file.
    :return: None
    """
    load = MonoLoader(filename=path, sampleRate=desired_fs)
    write = MonoWriter(filename=path, format="wav", sampleRate=desired_fs)
    write(load())


def resample_signal(signal, current_fs, desired_fs):
    """
    Re-samples the signal to 16 kHz for speech-related processing.
    Assumes the passed signal was obtained via a 44.1 kHz sampling rate
    :param signal: original signal
    :param current_fs: current sampling frequency of signal
    :param desired_fs: desired sampling frequency of signal
    :return: signal re-sampled at 16 kHz
    """
    converter = Resample(inputSampleRate=current_fs, outputSampleRate=desired_fs)
    return converter(signal)


def play_file(path_to_file, frame_size=1024):
    """
    Plays a WAV file through the standard output stream.
    :param path_to_file: path to the audio file
    :param frame_size: size of audio frame to output to each data request from the stream
    :return: None
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


def read_audio_file(path, desired_fs):
    """
    Reads in an audio file and returns an essentia (numpy float64) array of the data.
    :param path: path to the audio file
    :param desired_fs: desired sampling frequency of the returned signal
    :return: audio data and its sampling rate
    """
    # Currently only supports .wav files
    extension = environment.file_extension(path).lower()
    if extension != '.wav':
        raise NotImplementedError("read_audio_file() currently only supports .wav files, not %s" % extension)

    return np.array(MonoLoader(filename=path, sampleRate=desired_fs)())


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
        x = read_audio_file(signal, fs)
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

# python imports
from __future__ import print_function
import os
import pickle
import glob
# third-party application imports
from essentia.standard import Centroid, Flux, LPC, Windowing, ZeroCrossingRate, Energy, Entropy, RollOff, \
    SpectralPeaks, MFCC, FFT, Mean, EnergyBandRatio
import numpy as np
import pandas as pd
# application imports
from common.audio import read_audio_file, EPSILON

# static global members
WINDOW_SPECS = {
    # 'st_win': 0.048,
    # 'st_step': 0.016
    'st_win': 1,
    'st_step': 0.5
}
N_FFT = 1024
N_FORMANT = 4
N_LPC = 6
N_MFCC = 13
FS = 16000
# obtain pointers to essentia feature extraction functions
fft = FFT(size=N_FFT)
entropy = Entropy()
# lpc = LPC(order=N_LPC, sampleRate=FS)
formant_frequencies = SpectralPeaks(maxPeaks=N_FORMANT, sampleRate=FS)
energy = EnergyBandRatio(startFrequency=4000, stopFrequency=8000)
centroid = Centroid()
zcr = ZeroCrossingRate()
flux = Flux()
roll_off = RollOff(sampleRate=FS)
mfcc = MFCC(sampleRate=FS, inputSize=N_FFT, numberBands=40, numberCoefficients=N_MFCC,
            lowFrequencyBound=0, highFrequencyBound=8000)


def load_data(path):
    """
    Loads in data from the file specified, with only the columns specified by the
    column_indexes boolean mask. This supports the loading of custom features.
    :param path: Filename (no extension) where the model pickle is located
    :return: Loaded data set, indexed accordingly
    """
    with open(path, 'r') as fin:
        all_data = pickle.load(fin)
        features = all_data.iloc[:, :-1].as_matrix()
        labels = all_data.iloc[:, -1].as_matrix()
        frame_indices = pickle.load(fin)

    return features, labels, frame_indices


def save_data(path, features, labels, frame_indices):
    """
    Saves data to the file specified, with only the columns specified by the
    column_indexes boolean mask. This supports the saving of custom features.
    :param path: ilename (no extension) where the model pickle is located
    :param features: feature matrix
    :param labels: output labels for each entry in the matrix
    :param frame_indices: indices indicating the start of each frame
    """
    all_data = pd.DataFrame(features)
    labels = pd.DataFrame(labels, columns=['Class Label'])
    all_data = pd.concat((all_data, labels), axis=1)
    with open(path, 'w') as fout:
        pickle.dump(all_data, fout)
        pickle.dump(frame_indices, fout)


def normalize(features):
    """
    Normalizes a feature matrix to 0-mean and 1-std.
    :param features: raw numpy feature matrix
    :return: normalized numpy feature matrix
    """
    mean_data = np.mean(features, axis=0) + EPSILON
    std_data = np.std(features, axis=0) + EPSILON
    res = (features - mean_data) / std_data
    return res


def st_feature_extraction(signal, fs):
    """
    Extracts short-time features from a signal by framing.
    Enhanced from pyAudioAnalysis library https://github.com/tyiannak/pyAudioAnalysis.
    :param signal: signal to extract features from
    :param fs: digital sampling frequency
    :return: numpy feature array of shape (n_features x n_frames)
    """
    window_size = int(round(WINDOW_SPECS['st_win'] * fs))
    window_step = int(round(WINDOW_SPECS['st_step'] * fs))

    # Signal normalization
    signal /= (2.0 ** 15)
    dc_value = signal.mean()
    max_value = abs(signal).max()
    if max_value > 0:
        signal = (signal - dc_value) / max_value

    # counters
    signal_length = len(signal)
    cur_pos = 0
    frame_count = 0

    # compute the filter banks used for MFCC computation
    n_frames = int(signal_length / window_step)
    n_features = 22
    st_features = np.empty(shape=(n_frames, n_features), dtype=np.float64)

    # frame-by-frame processing of the signal
    while cur_pos + window_size - 1 < signal_length:
        # get current frame
        x = signal[cur_pos:cur_pos + window_size]
        # update window position
        cur_pos += window_step
        # normalized FFT magnitude
        windowed = Windowing(type="hamming", size=N_FFT)
        current_fft = abs(fft(windowed(x)))[:N_FFT] / N_FFT
        # extract features for current signal frame
        current_features = np.array([zcr(x)] +
                                    [energy(x)] +
                                    formant_frequencies(x)[0].tolist() +
                                    [centroid(x)] +
                                    [entropy(current_fft)] +
                                    [flux(current_fft)] +
                                    [roll_off(current_fft)] +
                                    mfcc(current_fft)[1][1:].tolist())
        # update feature matrix
        # TODO only compute the features necessary, according to the feature_mask.
        st_features[frame_count, :] = current_features
        frame_count += 1

    # zero out bad data points
    # TODO be smarter than this
    st_features[np.isnan(st_features) | np.isinf(st_features)] = 0.0
    return normalize(st_features)


def extract_from_file(filename):
    """
    Extracts the short-time features from the frames of a specified WAV file
    :param filename: WAV file path
    :return: features extracted from the file
    """
    fs = 16000
    x = read_audio_file(filename, fs)
    features = st_feature_extraction(x, fs)
    return features, fs


def extract_from_dir(directory):
    """
    Extracts features from a list of directories
    :param directory: list of directories to extract audio features from
    :return: tuple (4,)
        [0] feature matrix
        [1] unique class labels
        [2] per frame class labels
        [3] indices for the start of each frame
    """
    # pre-allocation
    wav_files = sorted(glob.glob(os.sep.join((directory, "*.wav"))))
    n_wav_files = len(wav_files)
    frame_feature_matrices = []
    class_labels = []
    frame_indices = []
    sample_index = 0

    # extract features from each file in the directory
    for i, wav_file in enumerate(wav_files):
        if i % 20 == 0:
            print("{}/{} in {}".format(i+1, n_wav_files, directory))
        [new_features, _] = extract_from_file(wav_file)
        frame_indices.append(sample_index)
        sample_index += new_features.shape[0]
        frame_feature_matrices.append(new_features)
        file_name = os.path.basename(wav_file)
        class_labels += [file_name[:file_name.find("_")]] * new_features.shape[0]
    frame_indices.append(sample_index)
    feature_matrix = np.vstack([m for m in frame_feature_matrices])
    return feature_matrix, np.unique(class_labels), class_labels, frame_indices

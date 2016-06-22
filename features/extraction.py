# python imports
from __future__ import print_function
import os
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
FEATURE_NAMES = np.array([
    'Short-Time ZCR',
    'Short-Time Energy',
    'Formant 1',
    'Formant 2',
    'Formant 3',
    'Formant 4',
    'Spectral Centroid',
    'Spectral Entropy',
    'Spectral Flux',
    'Spectral Roll-Off',
    'MFCC Coefficient 2',
    'MFCC Coefficient 3',
    'MFCC Coefficient 4',
    'MFCC Coefficient 5',
    'MFCC Coefficient 6',
    'MFCC Coefficient 7',
    'MFCC Coefficient 8',
    'MFCC Coefficient 9',
    'MFCC Coefficient 10',
    'MFCC Coefficient 11',
    'MFCC Coefficient 12',
    'MFCC Coefficient 13',
])
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


def load_data(model_name, column_indexes):
    """
    Loads in data from the file specified, with only the columns specified by the
    column_indexes boolean mask. This supports the loading of custom features.
    :param model_name: Filename (no extension) where the model pickle is located
    :param column_indexes: Boolean mask of matrix columns to include
    :return: Loaded data set, indexed accordingly
    """
    all_data = pd.read_pickle(model_name)
    features = all_data.iloc[:, :-1].as_matrix()
    features = features[:, column_indexes]
    labels = all_data.iloc[:, -1].as_matrix()
    return features, labels


def save_data(path, features, feature_names, labels):
    """
    Saves data to the file specified, with only the columns specified by the
    column_indexes boolean mask. This supports the saving of custom features.
    :param path: Filename (no extension) where the model pickle is located
    :param features: Feature matrix
    :param feature_names: Feature/column names for the data set
    :param labels: Output labels for each entry in the matrix
    """
    all_data = pd.DataFrame(features, columns=feature_names)
    labels = pd.DataFrame(labels, columns=['Class Label'])
    all_data = pd.concat((all_data, labels), axis=1)
    pd.to_pickle(all_data, path)


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


def st_feature_extraction(signal, fs, feature_mask):
    """
    Extracts short-time features from a signal by framing.
    Enhanced from pyAudioAnalysis library https://github.com/tyiannak/pyAudioAnalysis.
    :param signal: signal to extract features from
    :param fs: digital sampling frequency
    :param feature_mask: boolean mask for feature selection
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
    n_features = len(feature_mask[feature_mask])
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
        st_features[frame_count, :] = current_features[feature_mask]
        frame_count += 1

    # TODO only compute the features necessary, according to the feature_mask.
    # Indexing afterwards is just a quick and dirty way to get the functionality.
    st_features[np.isnan(st_features) | np.isinf(st_features)] = 0.0
    return normalize(st_features)


def extract_from_file(filename, feature_mask):
    """
    Extracts the short-time features from the frames of a specified WAV file
    :param filename: WAV file path
    :param feature_mask: boolean mask to use for feature selection
    :return: features extracted from the file
    """
    [fs, x] = read_audio_file(filename)
    features = st_feature_extraction(x, fs, feature_mask)
    return features, fs


def extract_from_dir(directory, feature_mask):
    """
    Extracts features from a list of directories
    :param directory: list of directories to extract audio features from
    :param feature_mask: boolean mask for feature selection
    :return:
    """
    # pre-allocation
    wav_files = sorted(glob.glob(os.sep.join((directory, "*.wav"))))
    n_wav_files = len(wav_files)
    feature_matrices = []
    class_labels = []
    sample_indices = []
    sample_index = 0

    # extract features from each file in the directory
    for i, wav_file in enumerate(wav_files):
        if i % 20 == 0:
            print("{}/{} in {}".format(i+1, n_wav_files, directory))
        [new_features, _] = extract_from_file(wav_file, feature_mask)
        sample_indices.append(sample_index)
        sample_index += new_features.shape[0]
        feature_matrices.append(new_features)
        file_name = os.path.basename(wav_file)
        class_labels += [file_name[:file_name.find("_")]] * new_features.shape[0]
    sample_indices.append(sample_index)
    features = np.vstack([m for m in feature_matrices])
    return features, np.unique(class_labels), wav_files, class_labels, sample_indices

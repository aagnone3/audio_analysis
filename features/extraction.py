# Credit to pyAudioAnalysis for base code which led to the creation of this module.
# https://github.com/tyiannak/pyAudioAnalysis

# python imports
import os
# third-party application imports
import numpy as np
import pandas as pd
from audiolazy.lazy_lpc import lpc
# application imports
from common.audio import read_audio_file
from common.environment import files_with_extension
from .cepstral_domain import *
from .spectral_domain import *
from .time_domain import *


# static global members
WINDOW_SPECS = {
    'mt_win': 1.0,
    'mt_step': 1.0,
    'st_win': 0.048,
    'st_step': 0.016
}
FEATURE_NAMES = np.array([
    'Short-Time ZCR',
    'Short-Time Energy',
    'Short-Time Energy Entropy',
    'Formant 1',
    'Formant 2',
    'Formant 3',
    'Formant 4',
    'LSF 1',
    'LSF 2',
    'LSF 3',
    'LSF 4',
    'LSF 5',
    'LSF 6',
    'Spectral Centroid',
    'Spectral Spread',
    'Spectral Entropy',
    'Spectral Flux',
    'Spectral Roll-Off',
    'MFCC Coefficient 1',
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
    'Chroma A',
    'Chroma A#',
    'Chroma B',
    'Chroma C',
    'Chroma C#',
    'Chroma D',
    'Chroma D#',
    'Chroma E',
    'Chroma F',
    'Chroma F#',
    'Chroma G',
    'Chroma G#',
    'Chroma Feature Std Dev',
])


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
    features = normalize(features)
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
    This function normalizes a feature set to 0-mean and 1-std.
    Used in most classifier training cases.
    :param features: list of feature matrices (each one of them is a numpy matrix)
    :return: normalized numpy feature matrix
    """
    mean_data = np.mean(features, axis=0)
    std_data = np.std(features, axis=0)
    return np.array([(instance - mean_data) / std_data for instance in features])


def st_feature_extraction(signal, fs, window_size, window_step, feature_mask):
    """
    Extracts short-time features from a signal by framing.
    Enhanced from pyAudioAnalysis library https://github.com/tyiannak/pyAudioAnalysis.
    :param signal: signal to extract features from
    :param fs: digital sampling frequency
    :param window_size: size of the window
    :param window_step: shift amount of the window
    :param feature_mask: boolean mask for feature selection
    :return: numpy feature array of shape (n_features x n_frames)
    """

    window_size = int(window_size)
    window_step = int(window_step)

    # Signal normalization
    signal = np.double(signal)
    signal /= (2.0 ** 15)
    dc_value = signal.mean()
    max_value = (np.abs(signal)).max()
    if max_value > 0:
        signal = (signal - dc_value) / max_value

    # counters
    signal_length = len(signal)
    cur_pos = 0
    frame_count = 0
    n_fft = int(window_size / 2)

    # compute the filter banks used for MFCC computation
    [filter_bank, _] = mfcc_init_filter_banks(fs, n_fft)
    [n_chroma, n_freqs_per_chroma] = st_chroma_features_init(n_fft, fs)
    n_mfcc = 13
    st_features = np.array([], dtype=np.float64)
    previous_fft = []

    # frame-by-frame processing of the signal
    while cur_pos + window_size - 1 < signal_length:
        frame_count += 1
        # get current frame
        x = signal[cur_pos:cur_pos + window_size]
        # update window position
        cur_pos += window_step
        # FFT magnitude
        current_fft = abs(fft(x))
        # normalize the FFT magnitudes
        current_fft = current_fft[0:n_fft]
        current_fft /= len(current_fft)
        # keep previous FFT magnitude for spectral flux calculation
        if frame_count == 1:
            previous_fft = current_fft.copy()
        # setup current feature vector
        lpc_a, _ = lpc.autocor(x, 16)
        lpc_a = list(lpc_a.values())
        formants = formant_frequencies(x, fs, lpc_a)
        lsp = line_spectral_pairs(lpc_a)
        chroma_names, chroma_f = st_chroma_features(current_fft, fs, n_chroma, n_freqs_per_chroma)
        current_features_list = []
        # zero-crossing rate
        current_features_list.append(zcr(x))
        # signal energy
        current_features_list.append(energy(x))
        # energy entropy
        current_features_list.append(energy_entropy(x))
        # 1st formant frequency
        current_features_list.append(formants[0])
        # 2nd formant frequency
        current_features_list.append(formants[1])
        # 3rd formant frequency
        current_features_list.append(formants[2])
        # 4th formant frequency
        current_features_list.append(formants[3])
        # 1st line spectral pair
        current_features_list.append(lsp[0])
        # 2nd line spectral pair
        current_features_list.append(lsp[1])
        # 3rd line spectral pair
        current_features_list.append(lsp[2])
        # 4th line spectral pair
        current_features_list.append(lsp[3])
        # 5th line spectral pair
        current_features_list.append(lsp[4])
        # 6th line spectral pair
        current_features_list.append(lsp[5])
        # spectral centroid and spectral spread
        for elem in spectral_centroid_and_spread(current_fft, fs):
            if np.any([np.isnan(elem), np.isinf(elem)]):
                current_features_list.append(0.0)
            else:
                current_features_list.append(elem)
        # spectral entropy
        current_features_list.append(spectral_entropy(current_fft))
        # spectral flux
        current_features_list.append(spectral_flux(current_fft, previous_fft))
        # spectral roll-off
        current_features_list.append(spectral_roll_off(current_fft, 0.90, fs))
        # mfcc coefficients
        for mfcc in st_mfcc(current_fft, filter_bank, n_mfcc).copy():
            current_features_list.append(mfcc)
        # chroma features
        for f in chroma_f:
            if np.any([np.isnan(f), np.isinf(f)]):
                current_features_list.append(0.0)
            else:
                current_features_list.append(f)
        # standard deviation of chroma features
        std_chroma = chroma_f.std()
        if np.any([np.isnan(std_chroma), np.isinf(std_chroma)]):
            current_features_list.append(0.0)
        else:
            current_features_list.append(chroma_f.std())
        # form a numpy array of the current features
        current_features = np.empty((len(current_features_list), 1))
        current_features[:, 0] = current_features_list
        # initialize/update overall feature matrix
        if frame_count == 1:
            st_features = current_features
        else:
            st_features = np.concatenate((st_features, current_features), 1)
        previous_fft = current_fft.copy()

    # TODO only compute the features necessary, according to the feature_mask.
    # Indexing afterwards is just a quick and dirty way to get the functionality.
    # TODO is this transposed the wrong way?
    return np.array(st_features[feature_mask])


def mt_feature_extraction(signal, fs, feature_mask):
    """
    Averages the short-time features to form a crude approximation of the overall signal's features.
    :param signal: signal to analyze
    :param fs: sampling frequency
    :param feature_mask: boolean mask to use for feature selection
    :return: averaged features extracted from the signal
    """
    mt_win = round(WINDOW_SPECS['mt_win'] * fs)
    mt_step = round(WINDOW_SPECS['mt_step'] * fs)
    st_window_size = round(WINDOW_SPECS['st_win'] * fs)
    st_window_step = round(WINDOW_SPECS['st_step'] * fs)

    win_size_ratio = int(round(mt_win / st_window_step))
    win_step_ratio = int(round(mt_step / st_window_step))

    st_features = st_feature_extraction(signal, fs, st_window_size, st_window_step, feature_mask)
    num_features = len(st_features)
    # we have two choices for statistics here: mean and std of each feature
    num_statistics = 1
    mt_features = []
    for i in range(num_statistics * num_features):
        mt_features.append([])
    # use averaging to form mid-term features
    for i in range(num_features):
        cur_pos = 0
        N = len(st_features[i])
        while cur_pos < N:
            N1 = cur_pos
            N2 = cur_pos + win_size_ratio
            if N2 > N:
                N2 = N
            cur_st_features = st_features[i][N1:N2]

            mt_features[i].append(np.mean(cur_st_features))
            if num_statistics == 2:
                mt_features[i+num_features].append(np.std(cur_st_features))
            cur_pos += win_step_ratio

    mt_features = np.array(mt_features)
    return mt_features, st_features


def extract_from_file(filename, feature_mask):
    """
    Extracts the "mid-term" features from a specified WAV file
    :param filename: WAV file path
    :param feature_mask: boolean mask to use for feature selection
    :return: features extracted from the file
    """
    [fs, x] = read_audio_file(filename)
    [features, _] = mt_feature_extraction(x, fs, feature_mask)
    features = np.transpose(features)
    features = features.mean(axis=0)  # long term averaging of mid-term statistics
    return features, fs


def extract_from_dir(directory, feature_mask):
    """
    Extracts the "mid-term" feature of the WAV files in the specified directory
    :param directory: directory to work in
    :param feature_mask: boolean mask to use for feature selection
    :return: features extracted from the WAV files in the directory
    """

    # pre-allocate the feature matrix
    wav_files = sorted(files_with_extension(directory, ".wav"))
    mt_features = np.zeros((len(wav_files), len(feature_mask[feature_mask])))

    for i, wav_file in enumerate(wav_files):
        if i % 10 == 0:
            print(wav_file)
        # extract features from current file
        [new_mt_features, _] = extract_from_file(wav_file, feature_mask)
        # TODO is this reshape necessary?
        if len(new_mt_features.shape) == 1:
            new_mt_features = new_mt_features.reshape([1, -1])
        mt_features[i, :] = new_mt_features

    print('Features extracted from ' + directory)
    return np.array(mt_features), wav_files


# TODO do away with this, have all files in one directory, each name prefixed with the class name
# TODO then move any necessary functionality from this method into the extract_from_dir function
def extract_from_dirs(directories, feature_mask):
    """
    Extracts features from a list of directories
    :param directories: list of directories to extract audio features from
    :param feature_mask: boolean mask for feature selection
    :return:
    """
    # feature extraction for each class, where each directory represents samples for a class
    features = []
    class_names = []
    file_names = []
    class_labels = []
    for i, d in enumerate(directories):
        [f, fn] = extract_from_dir(d, feature_mask)
        num_files = len(fn)
        if f.shape[0] > 0:       # if at least one audio file has been found in the provided folder:
            features.append(f)
            file_names.append(fn)
            # Append class names
            if d[-1] == os.sep:
                class_name = d.split(os.sep)[-2]
            else:
                class_name = d.split(os.sep)[-1]
            class_names.append(class_name)
            # Append class labels for model accuracy testing
            for _ in range(num_files):
                class_labels.append(class_name)
    class_labels = np.array(class_labels)

    if len(features) == 0:
        raise Exception("Error: No data found in input folders for feature extraction!")

    # unravel the features by one dimension
    features = np.array(features)
    dim1 = features.shape[0] * features.shape[1]
    dim2 = features.shape[2]
    features = features.reshape(dim1, dim2)
    return features, class_names, file_names, class_labels

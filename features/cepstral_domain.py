import numpy as np
from numpy import log
from scipy.fftpack import fft, ifft
from scipy.fftpack.realtransforms import dct
from common.audio import EPSILON


def complex_cepstrum(frame, n_fft):
    """
    Computes the complex cepstrum of the FFT magnitude frame.
    :param frame: FFT magnitude frame
    :param n_fft: size of the FFT performed
    :return: complex cepstrum of the FFT magnitude frame
    """
    return ifft(log(fft(frame, n_fft)))


def real_cepstrum(frame, n_fft):
    """
    Computes the real cepstrum of the FFT magnitude frame.
    :param frame: FFT magnitude frame
    :param n_fft: size of the FFT performed
    :return: real cepstrum of the FFT magnitude frame
    """
    return np.real(complex_cepstrum(frame, n_fft))


# TODO custom MFCC module in C++
def mfcc_init_filter_banks(fs, n_fft):
    """
    Computes the triangular filter bank for MFCC computation.
    This function is copied from the scikits.talkbox library (MIT Licence)
    https://pypi.python.org/pypi/scikits.talkbox
    :param fs: Sampling frequency
    :param n_fft: FFT size
    :return: triangular filter banks for MFCC computation
    """
    # filter bank params
    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = np.zeros(nFiltTotal+2)
    freqs[:numLinFiltTotal] = lowfreq + np.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** np.arange(1, numLogFilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nFiltTotal, n_fft))
    nfreqs = np.arange(n_fft) / (1. * n_fft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i+1]
        highTrFreq = freqs[i+2]

        lid = np.arange(np.floor(lowTrFreq * n_fft / fs) + 1, np.floor(cenTrFreq * n_fft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = np.arange(np.floor(cenTrFreq * n_fft / fs) + 1, np.floor(highTrFreq * n_fft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def st_mfcc(X, fbank, nceps):
    """
    Computes the MFCCs of an FFT frame.
    Adapted from the scikits.talkbox (MIT license) implementation.
    :param X: FFT manitude frame
    :param fbank: filter bank created by mfcc_init_filter_banks()
    :param nceps: size of the desired cepstral vector
    :return: MFCC vector
    """
    mel_cepstrum = np.log10(np.dot(X, fbank.T) + EPSILON)
    return dct(mel_cepstrum, type=2, norm='ortho', axis=-1)[:nceps]


def st_chroma_features_init(n_fft, fs):
    """
    Initializes the chroma matrices used in chroma feature computations.
    Adapted from pyAudioAnalysis https://github.com/tyiannak/pyAudioAnalysis.
    :param n_fft: FFT size
    :param fs: sampling frequency
    :return: chroma matrices
    """

    frequencies = np.array([((f + 1) * fs) / (2 * n_fft) for f in range(n_fft)])
    Cp = 27.50

    n_chroma = np.round(12.0 * np.log2(frequencies / Cp)).astype(int)
    n_freqs_per_chroma = np.zeros((n_chroma.shape[0], ))
    unique_chroma = np.unique(n_chroma)

    for u in unique_chroma:
        idx = np.nonzero(n_chroma == u)
        n_freqs_per_chroma[idx] = idx[0].shape
    return n_chroma, n_freqs_per_chroma


def st_chroma_features(fft_frame, fs, n_chroma, n_freqs_per_chroma):
    """
    Computes the short-time chroma features of the FFT magnitude frame.
    Adapted from pyAudioAnalysis https://github.com/tyiannak/pyAudioAnalysis
    :param fft_frame: FFT magnitude frame
    :param fs: sampling frequency
    :param n_chroma: # chroma features
    :param n_freqs_per_chroma: # frequencies per chroma
    :return: Short-time chroma features
    """

    chroma_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    spectral_energy = fft_frame ** 2

    chroma = np.zeros((n_chroma.shape[0],))
    chroma[n_chroma] = spectral_energy
    chroma /= n_freqs_per_chroma[n_chroma]

    new_dimension = int(np.ceil(chroma.shape[0] / 12.0) * 12)
    temp_chroma = np.zeros((new_dimension, ))
    temp_chroma[:chroma.shape[0]] = chroma
    chroma = temp_chroma.reshape(temp_chroma.shape[0]/12, 12)

    chroma = np.matrix(np.sum(chroma, axis=0)).T
    chroma /= spectral_energy.sum()
    return chroma_names, chroma

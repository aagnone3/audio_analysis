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
    # filter bank params:
    lowfreq = 133.33
    linsc = 200 / 3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = np.zeros(nFiltTotal + 2)
    freqs[:numLinFiltTotal] = lowfreq + np.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal - 1] * logsc ** np.arange(1, numLogFilt + 3)
    heights = 2. / (freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nFiltTotal, n_fft))
    nfreqs = np.arange(n_fft) / (1. * n_fft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i + 1]
        highTrFreq = freqs[i + 2]

        lid = np.arange(np.floor(lowTrFreq * n_fft / fs) + 1, np.floor(cenTrFreq * n_fft / fs) + 1,
                        dtype=np.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = np.arange(np.floor(cenTrFreq * n_fft / fs) + 1, np.floor(highTrFreq * n_fft / fs) + 1,
                        dtype=np.int)
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
    mspec = np.log10(np.dot(X, fbank.T)+EPSILON)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps

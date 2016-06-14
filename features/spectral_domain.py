# Credit to pyAudioAnalysis for base code which led to the creation of this module.
# https://github.com/tyiannak/pyAudioAnalysis
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.signal import lfilter, deconvolve

from features.time_domain import zcr
from common import environment
from common.audio import EPSILON


def formant_frequencies(x, fs, lpc_coefficients):
    """
    Finds the formant frequencies of the signal represented by its LPC coefficients.
    :param x: original signal
    :param fs: digital sampling frequency
    :param lpc_coefficients: LPC coefficient vector of the original signal
    :return: formant frequencies of the signal
    """

    # apply hamming window and high-pass filter
    # x1 = x * np.hamming(len(x))
    # x1 = lfilter([1], [1., 0.63], x1)

    # obtain roots from LPC coefficients
    rts = np.roots(lpc_coefficients)
    rts = [r for r in rts if np.imag(r) >= 0]

    # convert LPC roots to angles
    angles = np.arctan2(np.imag(rts), np.real(rts))

    # convert angles to frequencies
    frequencies = angles * (fs / (2 * math.pi))
    return sorted(frequencies[frequencies > 0])


def line_spectral_pairs(lpc_coefficients):
    """
    Computes the line spectral pairs from the lpc polynomial.
    Adapted from the poly2lsf.m MATLAB function.
    :param lpc_coefficients: LPC polynomial coefficients
    :return: vector of LSP coefficients
    """
    # Coerce to numpy array type and normalize
    lpc_coefficients = np.array(lpc_coefficients)
    if lpc_coefficients[0] != 1.0:
        lpc_coefficients /= 1.0
    # Form the sum and difference filters
    p = len(lpc_coefficients) - 1
    a1 = np.append(lpc_coefficients, np.array([0]))
    a2 = list(reversed(a1))
    sum_filter = a1 + a2
    diff_filter = a1 - a2
    # Unique root removal, depending on the order of the LPC coefficient vector
    if p % 2 != 0:
        # Remove z=1 and z=-1 root of difference filter
        [P, _] = deconvolve(diff_filter, np.array([1, 0, -1]))
        Q = sum_filter
    else:
        # Remove the z=1 diff_filter root, and the z=-1 sum filter root
        [P, _] = deconvolve(diff_filter, np.array([1, -1]))
        [Q, _] = deconvolve(sum_filter, np.array([1, 1]))

    roots_p = np.roots(P)
    roots_q = np.roots(Q)

    angles_p = np.angle(roots_p[0::2])
    angles_q = np.angle(roots_q[0::2])

    ls_pairs = np.sort(np.concatenate([angles_p, angles_q]))
    return ls_pairs


def spectral_centroid_and_spread(fft_frame, fs):
    """
    Computes spectral centroid and spread of an FFT magnitude frame.
    :param fft_frame: FFT magnitude frame
    :param fs: digital sampling frequency, in Hz
    :return: spectral centroid and spread of the FFT magnitude frame
    """

    ind = (np.arange(1, len(fft_frame) + 1)) * (fs / (2.0 * len(fft_frame)))
    frame = fft_frame.copy() / fft_frame.max()
    # compute centroid and spread
    num = np.sum(ind * frame)
    den = np.sum(frame) + EPSILON
    centroid = (num / den)
    spread = np.sqrt(np.sum(((ind - centroid) ** 2) * frame) / den)
    # normalize
    centroid /= (fs / 2.0)
    spread /= (fs / 2.0)

    return centroid, spread


def spectral_entropy(fft_frame, n_short_blocks=10):
    """
    Computes the spectral entropy of an FFT magnitude frame.
    :param fft_frame: FFT magnitude frame.
    :param n_short_blocks: sampling frequency, in Hz.
    :return: spectral entropy of the FFT magnitude frame.
    """
    frame_length = len(fft_frame)
    # Compute the normalized power spectral density (PSD)
    psd = np.sum(fft_frame ** 2) / (frame_length + EPSILON)
    # Return the entropy of the normalized PSD
    return - np.sum(psd * np.log2(psd + EPSILON))


def spectral_flux(current_frame, previous_frame):
    """
    Computes the spectral flux of an FFT magnitude frame.
    In this implementation, we consider the flux as the SSE of the normalized distances
    between the two frames.
    :param current_frame: current FFT magnitude frame.
    :param previous_frame: previous FFT magnitude frame.
    :return: spectral flux of an FFT magnitude frame.
    """
    # normalize frames
    current_frame /= np.sum(current_frame + EPSILON)
    previous_frame /= np.sum(previous_frame + EPSILON)
    # return the spectral flux, as defined in the documentation
    return np.sum((current_frame - previous_frame) ** 2)


def spectral_roll_off(fft_frame, c, fs):
    """
    Computes the spectral roll-off of an FFT magnitude frame.
    The spectral roll-off is defined as the frequency at which the spectral energy is equal to
    a threshold energy, as controlled by the c parameter.
    :param fft_frame: FFT magnitude frame
    :param c: constant
    :param fs: sampling frequency, in Hz.
    :return: spectral roll-of
    """
    n_fft = float(len(fft_frame))
    threshold = c * np.sum(fft_frame ** 2)
    # Compute the cumulative sum of the spectral energy
    cum_sum = np.cumsum(fft_frame ** 2) + EPSILON
    [a, ] = np.nonzero(cum_sum > threshold)
    if len(a) > 0:
        roll_off = np.float64(a[0]) / n_fft
    else:
        roll_off = 0.0
    return roll_off


def harmonic_ratio_and_pitch(frame, fs):
    """
    Computes the harmonic ratio and pitch of the given FFT magnitude frame
    :param frame: FFT magnitude frame
    :param fs: digital sampling frequency
    :return: harmonic ratio and pitch of the given FFT magnitude frame
    """
    M = np.round(0.016 * fs) - 1
    R = np.correlate(frame, frame, mode='full')

    g = R[len(frame)-1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = np.nonzero(np.diff(np.sign(R)))

    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

    gamma = np.zeros((M), dtype=np.float64)
    cum_sum = np.cumsum(frame ** 2)
    gamma[m0:M] = R[m0:M] / (np.sqrt((g * cum_sum[M:m0:-1])) + EPSILON)

    if zcr(gamma) > 0.15:
        harmonic_ratio = 0.0
        fundamental_frequency = 0.0
    else:
        if len(gamma) == 0:
            harmonic_ratio = 1.0
            blag = 0.0
            gamma = np.zeros(M, dtype=np.float64)
        else:
            harmonic_ratio = np.max(gamma)
            blag = np.argmax(gamma)

        # find fundamental frequency
        fundamental_frequency = fs / (blag + EPSILON)
        if fundamental_frequency > 5000:
            fundamental_frequency = 0.0
        if harmonic_ratio < 0.1:
            fundamental_frequency = 0.0

    return harmonic_ratio, fundamental_frequency

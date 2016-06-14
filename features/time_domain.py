# Credit to pyAudioAnalysis for base code which led to the creation of this module.
# https://github.com/tyiannak/pyAudioAnalysis
import numpy as np
from common.audio import EPSILON


def zcr(frame):
    """
    Computes the zero-crossing rate of a signal frame
    :param frame: signal frame, assumed to be properly windowed or otherwise pre-emphasized
    :return: the zero-crossing rate of the frame
    """
    sign_change_mask = np.abs(np.diff(np.sign(frame))) / 2
    num_crossings = np.sum(sign_change_mask)
    return np.float64(num_crossings) / np.float64(len(frame))


def energy(frame):
    """
    Computes the short-time energy of a signal frame.
    :param frame: signal frame, assumed to be properly windowed or otherwise pre-emphasized
    :return: the energy of the frame
    """
    return np.sum(frame ** 2) / np.float64(len(frame))


def energy_entropy(frame, num_short_blocks=10):
    """
    Computes the entropy of the short-time total_energy of a signal frame
    :param frame: signal frame, assumed to be properly windowed or otherwise pre-emphasized
    :param num_short_blocks:
    :return: entropy of the short-time total_energy of a signal frame
    """
    total_energy = np.sum(frame ** 2)
    frame_size = len(frame)
    sub_frame_size = int(np.floor(frame_size / num_short_blocks))
    if frame_size != sub_frame_size * num_short_blocks:
            frame = frame[0:sub_frame_size * num_short_blocks]
    # sub_frames is of size [numOfShortBlocks x frame_size]
    sub_frames = frame.reshape(sub_frame_size, num_short_blocks, order='F').copy()

    # compute the total_energy of each normalized sub-frame
    s = np.sum(sub_frames ** 2, axis=0) / (total_energy + EPSILON)

    # return entropy of the normalized sub-frame energies
    return -np.sum(s * np.log2(s + EPSILON))

"""Utils for the rifsstatemachine package"""

import numpy as np
import soundfile as sf

from math import log10


def save_wav(signal: np.array, target_path: str) -> None:
    """
    Save a numpy array as a .wav file.

    Parameters
    ----------
    signal : np.ndarray
        The signal to save
    target_path : str
        The path to save the signal to as a .wav file

    Returns
    -------
    None
    """
    sf.write(target_path, np.squeeze(signal), 16000)


def calculate_rms(data: np.ndarray) -> float:
    """Calculate RMS of the data
    Parameters
    ----------
    data : np.ndarray
        The data to calculate the RMS of
    Returns
    -------
    float
        The RMS of the data
    """
    return np.sqrt(np.mean(np.square(data)))


def calculate_dB(rms: float) -> float:
    """Calculate dB of the RMS
    Parameters
    ----------
    rms : float
        The RMS to calculate the dB of
    Returns
    -------
    float
        The dB of the RMS
    """
    if rms > 0:
        return 20 * log10(rms)
    return -100.0

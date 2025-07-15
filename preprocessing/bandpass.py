# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 11:30:30 2025

@author: milad
"""

import numpy as np
from scipy.signal import butter, filtfilt


class preprocess:
    """
    A collection of ECG preprocessing utilities including bandpass filtering and normalization.
    """

    def __init__(self, fs: float = 250.0, lowcut: float = 0.5, highcut: float = 40.0, order: int = 5):
        """
        :param fs: Sampling frequency of the ECG signal
        :param lowcut: Lower bound of bandpass filter (Hz)
        :param highcut: Upper bound of bandpass filter (Hz)
        :param order: Order of the Butterworth filter
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth bandpass filter to the ECG signal.
        """
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize the ECG signal to zero mean and unit variance.
        """
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing: bandpass + normalization.
        """
        filtered = self.bandpass_filter(signal)
        return self.normalize(filtered)
    

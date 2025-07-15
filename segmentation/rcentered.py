# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:24:39 2025

@author: milad
"""

import numpy as np
import neurokit2 as nk


class RCentered:
    """
    Class for segmenting ECG signals centered around R-peaks.
    """

    def __init__(self, fs: int = 250, segment_length_sec: float = 0.5):
        """
        :param fs: Sampling frequency of the ECG signal
        :param segment_length_sec: Length of each segment in seconds (centered on R-peak)
        """
        self.fs = fs
        self.segment_length_sec = segment_length_sec
        self.segment_length = int(self.segment_length_sec * self.fs)
        self.half_len = self.segment_length // 2

    def segment(self, signal: np.ndarray) -> np.ndarray:
        """
        Segment the ECG signal centered around detected R-peaks.

        :param signal: 1D ECG signal
        :return: Array of heartbeat segments
        """
        try:
            processed = nk.ecg_process(signal, sampling_rate=self.fs)
            r_peaks = processed[1]['ECG_R_Peaks']
        except Exception as e:
            print("R-peak detection failed:", e)
            return []

        segments = []
        for r in r_peaks:
            start = r - self.half_len
            end = r + self.half_len
            if start >= 0 and end <= len(signal):
                segments.append(signal[start:end])

        return np.array(segments)


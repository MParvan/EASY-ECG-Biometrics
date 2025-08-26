
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
import warnings


class Preprocess:
    def __init__(self, dataset_name: str = "Heartprint", normalize: bool = True, config: dict = None):
        """
        :param dataset_name: Name of the dataset to extract fs from config.
        :param normalize: Whether to apply normalization after filtering.
        :param config: Dictionary containing sampling frequencies for datasets.
        """
        if config is None:
            raise ValueError("You must provide a config dictionary with sampling frequencies.")

        if dataset_name not in config or "fs" not in config[dataset_name]:
            raise ValueError(f"Sampling frequency not found in config for dataset '{dataset_name}'.")

        self.fs = config[dataset_name]["fs"]
        self.normalize_flag = normalize

    def butter_filter(self, signal, order, cutoff, btype='band', highcut=None):
        nyquist = 0.5 * self.fs
        if btype == 'band':
            if highcut is None:
                raise ValueError("For bandpass, both lowcut and highcut must be provided.")
            low = cutoff / nyquist
            high = highcut / nyquist
            if high >= 1 or low >= 1:
                raise ValueError(f"Bandpass cutoff exceeds Nyquist ({nyquist}). Reduce highcut.")
            b, a = butter(order, [low, high], btype='band')
        else:
            freq = cutoff / nyquist
            if freq >= 1:
                raise ValueError(f"Cutoff exceeds Nyquist ({nyquist}). Lower the cutoff.")
            b, a = butter(order, freq, btype=btype)
        return filtfilt(b, a, signal)

    def median_filter(self, signal, kernel_size):
        if kernel_size % 2 == 0:
            raise ValueError("Median kernel size must be odd.")
        return medfilt(signal, kernel_size)

    def normalize(self, signal):
        std = np.std(signal)
        return (signal - np.mean(signal)) / (std + 1e-6) if std > 0 else signal

    def apply(self, signal: np.ndarray, method: str = "bandpass5-0.5-40"):
        """
        Parses a method string (e.g., 'bandpass5-0.5-40') and applies preprocessing.
        """
        method = method.lower()

        if method.startswith("bandpass"):
            parts = method.replace("bandpass", "").split("-")
            order = int(parts[0]) if len(parts) > 0 and parts[0] else 5
            lowcut = float(parts[1]) if len(parts) > 1 else 0.5
            highcut = float(parts[2]) if len(parts) > 2 else 40.0
            signal = self.butter_filter(signal, order, lowcut, btype="band", highcut=highcut)

        elif method.startswith("highpass"):
            parts = method.replace("highpass", "").split("-")
            order = int(parts[0]) if len(parts) > 0 else 5
            cutoff = float(parts[1]) if len(parts) > 1 else 0.5
            signal = self.butter_filter(signal, order, cutoff, btype="high")

        elif method.startswith("lowpass"):
            parts = method.replace("lowpass", "").split("-")
            order = int(parts[0]) if len(parts) > 0 else 5
            cutoff = float(parts[1]) if len(parts) > 1 else 40.0
            signal = self.butter_filter(signal, order, cutoff, btype="low")

        elif method.startswith("median"):
            parts = method.replace("median", "").split("-")
            kernel = int(parts[0]) if parts[0] else 5
            signal = self.median_filter(signal, kernel_size=kernel)

        else:
            raise NotImplementedError(f"Unknown preprocessing method: {method_str}")

        return self.normalize(signal) if self.normalize_flag else signal


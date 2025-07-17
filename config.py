# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 11:44:34 2025

@author: milad
"""

# config.py

CONFIG = {
    "Heartprint": {
        "fs": 250.0,           # Sampling frequency in Hz
        "sample_length": 3747  # Number of samples to use from each recording
    },
    "ECG-ID": {
        "fs": 500.0,
        "sample_length": 5000
    },
    "MIT-BIH": {
        "fs": 360.0,
        "sample_length": 6500
    },
    # Add more datasets as needed
}

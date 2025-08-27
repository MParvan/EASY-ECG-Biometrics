
import sys
sys.path.insert(1, 'C:\\Users\\milad\\Desktop\\ecg-biometrics')

import os
import numpy as np
import config
from config import CONFIG
from datasets.heartprint import HeartPrintDataset
from preprocessing.filters import Preprocess
from segmentation.rcentered import RCentered

datasets_path = os.path.join(os.getcwd(), "datasets")
data_path = os.path.join(datasets_path, "Heartprint")
FS = CONFIG["Heartprint"]["fs"]
SAMPLE_LENGTH = CONFIG["Heartprint"]["sample_length"]
dataset = HeartPrintDataset.get_all_recordings(data_path)

def load_data_experiment1():
    X, y = [], []
    train_sessions = ["Session-1"]
    for session in train_sessions:
        print(f"Processing session: {session}")
        if session not in dataset:
            print(f"Session {session} not found in dataset.")
            return

        for subject_id, files in dataset[session].items():
            for file in files:
                try:
                    with open(file, 'r') as f:
                        signal = [float(line.strip()) for line in f.readlines()[:SAMPLE_LENGTH]]
                        signal = np.array(signal)
                        preprocessing_method = "bandpass4-0.25-40"

                        pp = Preprocess(dataset_name="Heartprint", config=CONFIG)
                        signal_P = pp.apply(signal, method=preprocessing_method)
                        
                    ss = RCentered()    
                    segments = ss.segment(signal_P)
                    for seg in segments:
                        X.append(seg)
                        y.append(int(subject_id))
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        X = np.array(X)
        y = np.array(y)
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    return X, y


X, y = load_data_experiment1()
print(f"Loaded training data shape: {X.shape}, Labels shape: {y.shape}")
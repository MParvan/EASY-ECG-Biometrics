# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 12:04:41 2025

@author: milad
"""

import numpy as np
from datasets.ecg_id import ECGIDDataset
from datasets.heartprint import HeartPrintDataset
# from preprocessing.filters import bandpass_filter, normalize
# from segmentation.simple import sliding_window
# from models.identification_cnn import CNN1DIdentifier
from torch.utils.data import DataLoader
import torch
import os
from config import FS, SAMPLE_LENGTH
from sklearn.model_selection import train_test_split

from segmentation.rcentered import RCentered
from utils.utils import ECGUtils
from models.deepecg import DeepECGModel
from models.siamese import SiameseECGModel

# # Example config
# # DATA_PATH = "datasets/ECG-ID/"
# DATA_PATH = "datasets/Heartprint/"
# TASK = "identification"
# SEGMENT = True
# BATCH_SIZE = 32
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Load dataset
# # dataset = ECGIDDataset(
# #     path=DATA_PATH,
# #     task=TASK
# #     # segment_fn=sliding_window if SEGMENT else None,
# #     # transform=lambda x: normalize(bandpass_filter(x))
# # )

# dataset = HeartPrintDataset(DATA_PATH)

# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# # Initialize model
# # model = CNN1DIdentifier(num_classes=23).to(DEVICE)

# # [Training loop would go here...]


import argparse

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--Data", type=str, default="Heartprint",
                        help="dataset")
    
    parser.add_argument("--train_sessions", nargs='+', default=["Session-1"],
                        help="sessions for training (e.g. --train_sessions Session-1 Session-2)")

    parser.add_argument("--test_sessions", nargs='+', default=["Session-3L"],
                        help="sessions for testing (e.g. --test_sessions Session-3L)")
    
    parser.add_argument("--preprocessing", type=str, default="bandpass",
                        choices=["bandpass", "median", "CGAN"],
                        help="choose an available preprocessing method")
    
    parser.add_argument("--task", type=str, default="identification",
                        choices=["identification", "verification"],
                        help="choose a recognition task")


    
    
    args = parser.parse_args()

    return args


def main():
    args = get_parser()

    datasets_path = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(datasets_path, args.Data)
    dataset = HeartPrintDataset.get_all_recordings(data_path)
    
    X, y = [], []
    train_sessions = args.train_sessions
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
                        preprocessing_method = args.preprocessing
                        if preprocessing_method=="bandpass":
                            from preprocessing.bandpass import preprocess
                            pp = preprocess()
                            signal_P = pp.preprocess(signal)
                        
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

    
    X_test, y_test = [], []
    test_sessions = args.test_sessions
    
    for session in test_sessions:
        print(f"Processing test session: {session}")
        if session not in dataset:
            print(f"Session {session} not found in dataset.")
            continue  # Skip this session instead of returning
    
        for subject_id, files in dataset[session].items():
            for file in files:
                try:
                    with open(file, 'r') as f:
                        signal = [float(line.strip()) for line in f.readlines()[:SAMPLE_LENGTH]]
                        signal = np.array(signal)
    
                    # Preprocessing
                    preprocessing_method = args.preprocessing
                    if preprocessing_method == "bandpass":
                        from preprocessing.bandpass import preprocess
                        pp = preprocess()
                        signal_P = pp.preprocess(signal)
                    else:
                        signal_P = signal
    
                    # Segmentation
                    ss = RCentered()
                    segments = ss.segment(signal_P)
    
                    for seg in segments:
                        X_test.append(seg)
                        y_test.append(int(subject_id))
                        
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    X_test = np.array(X_test)[..., np.newaxis]
    y_test = np.array(y_test)


    utils = ECGUtils(fs=FS)
    task = args.task
    if task == "identification":
        # Split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

        # Build and compile model
        model_builder = DeepECGModel(input_shape=X.shape[1:], num_classes=len(y))
        model = model_builder.build()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train
        model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

        # Evaluate
        utils.evaluate_identification(model, X_test, y_test)

    elif task == "verification":
        # Create pairs from training data
        pairs, labels = utils.create_pairs(X, y)
        X1 = np.array([p[0] for p in pairs])
        X2 = np.array([p[1] for p in pairs])
        y_labels = np.array(labels)

        # Split into train/test
        X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
            X1, X2, y_labels, test_size=0.2, stratify=y_labels
        )

        # Build Siamese model
        model_builder = SiameseECGModel(input_shape=X.shape[1:])
        model = model_builder.build()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train
        model.fit([X1_train, X2_train], y_train, epochs=10, batch_size=32, validation_data=([X1_val, X2_val], y_val))

        # If you also want to evaluate on unseen session:
        test_pairs, test_labels = utils.create_pairs(X_test, y_test)
        X1_test = np.array([p[0] for p in test_pairs])
        X2_test = np.array([p[1] for p in test_pairs])
        y_test = np.array(test_labels)

        utils.evaluate_verification(model, X1_test, X2_test, y_test)
    


    
if __name__ == '__main__':
    main()


















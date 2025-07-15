# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:52:59 2025

@author: milad
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score


class ECGUtils:
    def __init__(self, fs=250):
        self.fs = fs

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def plot_signal(self, signal, title="ECG Signal"):
        time = np.arange(len(signal)) / self.fs
        plt.figure(figsize=(10, 3))
        plt.plot(time, signal)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def create_pairs(self, X, y):
        pairs = []
        labels = []
        class_indices = {}
        for idx, label in enumerate(y):
            class_indices.setdefault(label, []).append(idx)

        for label, indices in class_indices.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    # Positive pair
                    pairs.append((X[indices[i]], X[indices[j]]))
                    labels.append(1)

                    # Negative pair
                    neg_label = label
                    while neg_label == label:
                        neg_label = random.choice(list(class_indices.keys()))
                    neg_idx = random.choice(class_indices[neg_label])
                    pairs.append((X[indices[i]], X[neg_idx]))
                    labels.append(0)

        return np.array(pairs), np.array(labels)

    def evaluate_identification(self, model, X_test, y_test):
        y_pred = model.predict(X_test).argmax(axis=1)
        print("\n[Identification] Evaluation Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    def evaluate_verification(self, model, X1_test, X2_test, y_test):
        y_scores = model.predict([X1_test, X2_test]).flatten()
        y_pred = (y_scores > 0.5).astype(int)
        print("\n[Verification] Evaluation Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_scores):.4f}")

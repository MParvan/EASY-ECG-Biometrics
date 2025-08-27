
import tensorflow as tf
from tensorflow.keras import layers, models


class DeepECGModel:
    """
    Class to build a DeepECG model for identification tasks.
    """

    def __init__(self, input_shape, num_classes):
        """
        :param input_shape: Tuple, e.g., (250, 1) for 1-second ECG with 250Hz
        :param num_classes: Number of output classes (subjects)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        """
        Constructs and returns the DeepECG Keras model.
        """
        model = models.Sequential([
            layers.Conv1D(16, kernel_size=7, activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(32, kernel_size=5, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model

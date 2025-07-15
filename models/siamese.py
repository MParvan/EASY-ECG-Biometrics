# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:45:06 2025

@author: milad
"""

import tensorflow as tf
from tensorflow.keras import layers


class SiameseECGModel:
    """
    Class to build a Siamese network for ECG verification tasks.
    """

    def __init__(self, input_shape):
        """
        :param input_shape: Tuple, e.g., (250, 1)
        """
        self.input_shape = input_shape

    def build_base_network(self):
        """
        Constructs the shared base CNN model for feature extraction.
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        x = layers.Conv1D(32, 7, activation='relu')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        return tf.keras.Model(inputs, x)

    def build(self):
        """
        Constructs and returns the Siamese Keras model.
        """
        input_a = tf.keras.Input(shape=self.input_shape)
        input_b = tf.keras.Input(shape=self.input_shape)

        base_network = self.build_base_network()

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
        output = layers.Dense(1, activation='sigmoid')(distance)

        siamese_model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
        return siamese_model

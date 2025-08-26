
"""
Multi-Branch Network with Attention-Enhanced Residual Blocks for ECG Biometrics
Adapted from: ECG Biometrics Based on Attention-Enhanced Domain Adaptive Feature Fusion Network
"""

import tensorflow as tf
from tensorflow.keras import layers, models

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(channels // 8, activation='relu')
        self.dense2 = layers.Dense(channels, activation='sigmoid')
        self.multiply = layers.Multiply()

    def call(self, inputs):
        attention = self.global_pool(inputs)
        attention = self.dense1(attention)
        attention = self.dense2(attention)
        attention = tf.expand_dims(attention, axis=1)
        return self.multiply([inputs, attention])

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv1D(filters, kernel_size, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.att = AttentionBlock(filters)
        self.shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')
        self.add = layers.Add()

    def call(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.att(out)
        out = self.add([out, identity])
        return self.relu(out)

class MultiBranchECGModel:
    """
    Multi-branch residual attention model for ECG biometric identification.
    """

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape  # (length, channels)
        self.num_classes = num_classes

    def build_branch(self, kernel_size):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = ResidualBlock(filters=32, kernel_size=kernel_size)(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = ResidualBlock(filters=64, kernel_size=kernel_size)(x)
        x = layers.GlobalAveragePooling1D()(x)
        return models.Model(inputs, x)

    def build(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # Branches with different receptive fields
        branch3 = self.build_branch(kernel_size=3)(inputs)
        branch5 = self.build_branch(kernel_size=5)(inputs)
        branch7 = self.build_branch(kernel_size=7)(inputs)

        # Feature fusion
        merged = layers.Concatenate()([branch3, branch5, branch7])
        fused = layers.Dense(128, activation='relu')(merged)
        outputs = layers.Dense(self.num_classes, activation='softmax')(fused)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

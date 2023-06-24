import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Normalization, InputLayer, Dense, Dropout, BatchNormalization, LeakyReLU


def food_category_classification_model(X_train, n, dim=300):
    """
        Build neural network architecture for food category classification model

        Args:
            X_train: Training dataset for word embeddings
            n: Number of food categories
            dim: Dimension of word embeddings

        Returns:
            classification_model: model architecture
    """
    # Set seed
    np.random.seed(2023)
    # Feature Normalization
    embedding_normalizer = Normalization(input_shape=[dim, ], axis=None)
    embedding_normalizer.adapt(X_train)
    # Classification model architecture #Best parameter from gridsearch
    classification_model = Sequential([
        embedding_normalizer,
        Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(n, activation='softmax')
    ])

    return classification_model


def nutrition_prediction_model(dim=300):
    """
        Build neural network architecture for nutrition prediction model

        Args:
            dim: Dimension of word embeddings

        Returns:
            prediction_model: model architecture
    """
    # Set seed
    np.random.seed(2023)
    # Prediction model architecture
    prediction_model = Sequential([
        tf.keras.Input(shape=(dim,)),
        Dense(256),
        LeakyReLU(0.1),
        Dropout(0.2),
        Dense(256),
        LeakyReLU(0.1),
        Dropout(0.2),
        Dense(256),
        LeakyReLU(0.1),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])

    return prediction_model

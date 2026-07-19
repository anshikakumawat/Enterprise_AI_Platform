import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.regularizers import l1, l2

from sklearn.ensemble import IsolationForest

"""LSTM model"""
def build_lstm(input_shape):
    """
    Build an LSTM model.
    """

    model = Sequential()

    model.add(
        LSTM(
            units=64,
            input_shape=input_shape,
            kernel_regularizer=l2(0.001)
        )
    )

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model

"""GRU model"""
def build_gru(input_shape):
    """
    Build a GRU model.
    """

    model = Sequential()

    model.add(
        GRU(
            units=64,
            input_shape=input_shape,
            kernel_regularizer=l2(0.001)
        )
    )

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model

"""anomaly detection"""
def anomaly_detection(data):
    """
    Detect anomalies using Isolation Forest.
    """

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    prediction = model.fit_predict(data)

    return prediction
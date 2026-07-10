import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout

def build_lstm(input_shape):
    model=Sequential([
        LSTM(128,return_sequences=True,input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32,activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam",loss="mse",metrics=["mae"])
    return model

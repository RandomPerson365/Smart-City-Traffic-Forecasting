from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU,Dense,Dropout

def build_gru(input_shape):
    model=Sequential([
        GRU(128,return_sequences=True,input_shape=input_shape),
        Dropout(0.2),
        GRU(64),
        Dense(1)
    ])
    model.compile(optimizer="adam",loss="mse",metrics=["mae"])
    return model

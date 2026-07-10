import numpy as np

def make_sequences(series,lookback=24):
    X,y=[],[]
    values=np.asarray(series)
    for i in range(lookback,len(values)):
        X.append(values[i-lookback:i])
        y.append(values[i])
    return np.array(X),np.array(y)

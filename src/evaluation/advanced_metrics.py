import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def mape(y_true,y_pred):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    return np.mean(np.abs((y_true-y_pred)/(y_true+1e-8)))*100

def evaluate_all(y_true,y_pred):
    return {
        "MAE":mean_absolute_error(y_true,y_pred),
        "RMSE":float(np.sqrt(mean_squared_error(y_true,y_pred))),
        "MAPE":mape(y_true,y_pred),
        "R2":r2_score(y_true,y_pred)
    }

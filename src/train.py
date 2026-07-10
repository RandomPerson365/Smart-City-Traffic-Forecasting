import pandas as pd
from src.model_factory import get_model
from src.evaluation import evaluate

def train_once(X_train,y_train,X_valid,y_valid,model_name="xgboost"):
    wrapper=get_model(model_name)
    wrapper.train(X_train,y_train)
    pred=wrapper.predict(X_valid)
    metrics=evaluate(y_valid,pred)
    return wrapper,pred,metrics

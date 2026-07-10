from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

def get_model(name:str):
    name=name.lower()
    if name=="random_forest":
        return RandomForestModel()
    if name=="xgboost":
        return XGBoostModel()
    raise ValueError(f"Unknown model: {name}")

from src.model_factory import get_model
from src.evaluation import evaluate

class Trainer:
    def __init__(self,model_name="xgboost"):
        self.wrapper=get_model(model_name)

    def fit(self,X_train,y_train):
        self.wrapper.train(X_train,y_train)

    def validate(self,X_valid,y_valid):
        pred=self.wrapper.predict(X_valid)
        return pred,evaluate(y_valid,pred)

    def save(self,path):
        self.wrapper.save(path)

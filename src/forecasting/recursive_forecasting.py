import pandas as pd

class RecursiveForecaster:
    """
    Skeleton recursive forecaster.
    Future milestone will automatically recompute lag features.
    """
    def __init__(self,model,feature_fn):
        self.model=model
        self.feature_fn=feature_fn

    def forecast(self,history,future,features):
        history=history.copy()
        preds=[]
        for _,row in future.iterrows():
            temp=pd.concat([history,pd.DataFrame([row])],ignore_index=True)
            temp=self.feature_fn(temp)
            x=temp.iloc[[-1]][features]
            pred=float(self.model.predict(x)[0])
            temp.loc[temp.index[-1],"Vehicles"]=pred
            history=temp
            preds.append(pred)
        return preds

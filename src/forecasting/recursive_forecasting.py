import pandas as pd
import numpy as np


class RecursiveForecaster:
    """
    Recursive traffic forecaster.

    Uses previous predictions to build lag features
    for future timestamps.
    """

    def __init__(self, model, feature_fn):

        self.model = model

        self.feature_fn = feature_fn

    # ---------------------------------------

    def forecast(self, history, future, features):

        history = history.copy()

        history = history.sort_values("DateTime")

        predictions = []

        for _, row in future.iterrows():

            row = row.to_dict()

            row["Vehicles"] = np.nan

            temp = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

            temp = self.feature_fn(temp)

            missing = [c for c in features if c not in temp.columns]

            if missing:

                raise ValueError(f"Missing features: {missing}")

            x = temp.loc[[temp.index[-1]], features]

            x = x.fillna(0)

            prediction = float(self.model.predict(x)[0])

            prediction = max(0, round(prediction))

            temp.loc[temp.index[-1], "Vehicles"] = prediction

            history = temp[history.columns]

            predictions.append(prediction)

        result = future.copy()
        result["Vehicles"] = predictions
        return result

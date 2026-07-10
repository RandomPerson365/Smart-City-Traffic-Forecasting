import pandas as pd
import numpy as np


class RecursiveForecaster:

    def __init__(self, model, feature_fn):

        self.model = model
        self.feature_fn = feature_fn

    # Replace ONLY this function ↓↓↓

    def forecast(self, history, future, features):

        history = history.copy()

        history = history.sort_values("DateTime")

        predictions = []

        for _, row in future.iterrows():

            row = row.to_dict()

            row["Vehicles"] = np.nan

            # Add next timestamp
            history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

            # Only keep the last 30 rows for feature generation
            window = history.tail(30).copy()

            # Generate features
            window = self.feature_fn(window)

            missing = [c for c in features if c not in window.columns]

            if missing:
                raise ValueError(f"Missing features: {missing}")

            x = window.loc[[window.index[-1]], features].fillna(0)

            prediction = float(self.model.predict(x)[0])

            prediction = max(0, round(prediction))

            # Save prediction into history
            history.loc[history.index[-1], "Vehicles"] = prediction

            predictions.append(prediction)

        return predictions

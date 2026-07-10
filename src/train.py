"""
train.py

Train multiple machine learning models for Smart City Traffic Forecasting.
Version 2.0
"""

from pathlib import Path
from datetime import datetime
import json

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ModelTrainer:

    def __init__(self):

        self.data_path = Path("data/processed/train_features.csv")

        self.model_dir = Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.report_dir = Path("outputs/reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.feature_columns = []

        self.best_model = None
        self.best_model_name = None
        self.best_metrics = None

    # --------------------------------------------------

    def load_data(self):

        print("\nLoading engineered dataset...")

        df = pd.read_csv(self.data_path)

        df["DateTime"] = pd.to_datetime(df["DateTime"])

        print(f"Loaded {len(df)} rows")

        return df

    # --------------------------------------------------

    def prepare_data(self, df):

        print("Preparing training data...")

        # Remove rows created because of lag features
        df = df.dropna().copy()

        # Chronological split
        df = df.sort_values("DateTime")

        split_index = int(len(df) * 0.80)

        train_df = df.iloc[:split_index]
        valid_df = df.iloc[split_index:]

        drop_columns = ["Vehicles", "DateTime"]

        X_train = train_df.drop(columns=drop_columns)
        y_train = train_df["Vehicles"]

        X_valid = valid_df.drop(columns=drop_columns)
        y_valid = valid_df["Vehicles"]

        # Save feature names for inference
        self.feature_columns = list(X_train.columns)

        joblib.dump(self.feature_columns, self.model_dir / "feature_columns.pkl")

        print(f"Training Features : {len(self.feature_columns)}")

        print(f"Training Samples : {len(X_train)}")

        print(f"Validation Samples : {len(X_valid)}")

        return (X_train, X_valid, y_train, y_valid)

    # --------------------------------------------------

    def evaluate_model(self, model, X_valid, y_valid):

        prediction = model.predict(X_valid)

        mae = mean_absolute_error(y_valid, prediction)

        rmse = mean_squared_error(y_valid, prediction) ** 0.5

        r2 = r2_score(y_valid, prediction)

        return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

    # --------------------------------------------------
    # Build Models
    # --------------------------------------------------

    def build_models(self):

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=300, random_state=42, n_jobs=-1
            ),
        }

        if XGBOOST_AVAILABLE:

            models["XGBoost"] = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror",
            )

        return models

    # --------------------------------------------------
    # Train All Models
    # --------------------------------------------------

    def train_models(self, X_train, y_train, X_valid, y_valid):

        print("\n")
        print("=" * 60)
        print("Training Machine Learning Models")
        print("=" * 60)

        models = self.build_models()

        results = []

        best_rmse = float("inf")

        for name, model in models.items():

            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)

            metrics = self.evaluate_model(model, X_valid, y_valid)

            results.append(
                {
                    "Model": name,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                }
            )

            print(metrics)

            if metrics["RMSE"] < best_rmse:

                best_rmse = metrics["RMSE"]

                self.best_model = model

                self.best_model_name = name

                self.best_metrics = metrics

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values("RMSE")

        print("\n")
        print("=" * 60)
        print("Model Comparison")
        print("=" * 60)

        print(results_df)

        return results_df

    # --------------------------------------------------
    # Save Training Artifacts
    # --------------------------------------------------

    def save_artifacts(self, results_df):

        # Save best model
        joblib.dump(self.best_model, self.model_dir / "best_model.pkl")

        # Save comparison table
        results_df.to_csv(self.report_dir / "model_results.csv", index=False)

        # Metadata
        metadata = {
            "best_model": self.best_model_name,
            "metrics": self.best_metrics,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns,
        }

        with open(self.model_dir / "training_metadata.json", "w") as f:

            json.dump(metadata, f, indent=4)

        # Simple text report
        with open(self.report_dir / "metrics_summary.txt", "w") as f:

            f.write("=" * 60 + "\n")

            f.write("SMART CITY TRAFFIC FORECASTING\n")

            f.write("=" * 60 + "\n\n")

            f.write(f"Best Model : {self.best_model_name}\n\n")

            for key, value in self.best_metrics.items():

                f.write(f"{key} : {value}\n")

            f.write("\n")

            f.write(f"Training Samples : {len(self.feature_columns)} features\n")

        print("\nTraining artifacts saved.")

    # --------------------------------------------------
    # Main Training Function
    # --------------------------------------------------

    def train(self):

        print("=" * 60)
        print("TRAINING PIPELINE")
        print("=" * 60)

        df = self.load_data()

        (X_train, X_valid, y_train, y_valid) = self.prepare_data(df)

        results_df = self.train_models(X_train, y_train, X_valid, y_valid)

        self.save_artifacts(results_df)

        print("\n")
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        print(f"\nBest Model : {self.best_model_name}")

        print(f"RMSE : {self.best_metrics['RMSE']}")

        print(f"R² : {self.best_metrics['R2']}")

        return results_df

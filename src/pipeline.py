"""
pipeline.py

End-to-End Smart City Traffic Forecasting Pipeline

This module orchestrates the complete workflow:
1. Data Preprocessing
2. Feature Engineering
3. Visualization
4. Model Training
5. Recursive Forecasting
6. Submission Generation
"""

from pathlib import Path
import joblib
import pandas as pd

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.visualization import Visualizer
from src.train import ModelTrainer

from src.forecasting.recursive_forecasting import RecursiveForecaster
from src.forecasting.submission import build_submission


class Pipeline:

    def __init__(self):

        self.root = Path(__file__).resolve().parent.parent

        self.processor = DataPreprocessor(
            train_path=self.root / "data/raw/train.csv",
            test_path=self.root / "data/raw/test.csv",
            processed_dir=self.root / "data/processed",
        )

        self.engineer = FeatureEngineer()

        self.visualizer = Visualizer()

        self.trainer = ModelTrainer()

    # --------------------------------------------------

    def preprocess(self):

        print("\nSTEP 1 : PREPROCESSING")

        train, test = self.processor.preprocess()

        return train, test

    # --------------------------------------------------

    def feature_engineering(self, train, test):

        print("\nSTEP 2 : FEATURE ENGINEERING")

        train, test = self.engineer.engineer(train, test)

        processed = self.root / "data/processed"

        train.to_csv(processed / "train_features.csv", index=False)

        test.to_csv(processed / "test_features.csv", index=False)

        return train, test

    # --------------------------------------------------

    def visualize(self, train):

        print("\nSTEP 3 : VISUALIZATION")

        self.visualizer.generate_all(train)

    # --------------------------------------------------

    def train_models(self):

        print("\nSTEP 4 : MODEL TRAINING")

        self.trainer.train()

    # --------------------------------------------------

    def forecast(self):

        print("\nSTEP 5 : FORECASTING")

        model = joblib.load(self.root / "models/best_model.pkl")

        features = joblib.load(self.root / "models/feature_columns.pkl")

        train = pd.read_csv(self.root / "data/processed/train_clean.csv")

        test = pd.read_csv(self.root / "data/raw/test.csv")

        train["DateTime"] = pd.to_datetime(train["DateTime"])

        test["DateTime"] = pd.to_datetime(test["DateTime"])

        forecaster = RecursiveForecaster(model=model, feature_fn=self.engineer.engineer)

        predictions = []

        for junction in sorted(test["Junction"].unique()):

            history = train[train["Junction"] == junction]

            future = test[test["Junction"] == junction]

            preds = forecaster.forecast(history, future, features)

            predictions.extend(preds)

        build_submission(test["ID"], predictions)

        print("Submission generated.")

    # --------------------------------------------------

    def run(self):

        print("=" * 70)
        print("SMART CITY TRAFFIC FORECASTING")
        print("=" * 70)

        train, test = self.preprocess()

        train, test = self.feature_engineering(train, test)

        self.visualize(train)

        self.train_models()

        self.forecast()

        print("\nPROJECT COMPLETED SUCCESSFULLY")

"""
feature_engineering.py

Feature Engineering Module
Smart City Traffic Forecasting

This module is shared by BOTH

1. Training
2. Recursive Forecasting

so every feature is generated exactly the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(self):

        self.lags = [1, 2, 3, 6, 12, 24]

        self.windows = [3, 6, 12, 24]

    # ==========================================================
    # DATETIME FEATURES
    # ==========================================================

    def add_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        dt = df["DateTime"]

        df["Year"] = dt.dt.year

        df["Month"] = dt.dt.month

        df["Quarter"] = dt.dt.quarter

        df["Week"] = dt.dt.isocalendar().week.astype(int)

        df["Day"] = dt.dt.day

        df["Hour"] = dt.dt.hour

        df["Minute"] = dt.dt.minute

        df["Weekday"] = dt.dt.weekday

        df["DayOfYear"] = dt.dt.dayofyear

        df["IsWeekend"] = (df["Weekday"] >= 5).astype(int)

        return df

    # ==========================================================
    # CYCLICAL FEATURES
    # ==========================================================

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)

        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)

        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

        df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)

        df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

        return df

    # ==========================================================
    # LAG FEATURES
    # ==========================================================

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        if "Vehicles" not in df.columns:
            return df

        for lag in self.lags:

            df[f"lag_{lag}"] = df.groupby("Junction")["Vehicles"].shift(lag)

        return df

    # ==========================================================
    # ROLLING FEATURES
    # ==========================================================

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        if "Vehicles" not in df.columns:
            return df

        for window in self.windows:

            rolling = df.groupby("Junction")["Vehicles"].rolling(window)

            df[f"rolling_mean_{window}"] = rolling.mean().reset_index(
                level=0, drop=True
            )

            df[f"rolling_std_{window}"] = rolling.std().reset_index(level=0, drop=True)

        return df

    # ==========================================================
    # EXPONENTIAL MOVING AVERAGE
    # ==========================================================

    def add_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        if "Vehicles" not in df.columns:
            return df

        for span in [12, 24]:

            df[f"ema_{span}"] = df.groupby("Junction")["Vehicles"].transform(
                lambda s: s.ewm(span=span, adjust=False).mean()
            )

        return df

    # ==========================================================
    # COMPLETE TRANSFORM
    # ==========================================================

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        if "DateTime" in df.columns:

            df["DateTime"] = pd.to_datetime(df["DateTime"])

        df = self.add_datetime_features(df)

        df = self.add_cyclical_features(df)

        df = self.add_lag_features(df)

        df = self.add_rolling_features(df)

        df = self.add_ema_features(df)

        return df

    # ==========================================================
    # TRAINING PIPELINE
    # ==========================================================

    def engineer(self, train: pd.DataFrame, test: pd.DataFrame):

        print("=" * 60)
        print("Creating Features")
        print("=" * 60)

        train = self.transform(train)

        test = self.transform(test)

        print("Feature Engineering Complete")

        print()

        print(train.head())

        return train, test

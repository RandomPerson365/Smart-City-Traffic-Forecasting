"""
feature_engineering.py

Creates time-based and statistical features for
traffic forecasting.

Author: Smart City Traffic Forecasting Project
"""

import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(self):
        pass

    # -----------------------------
    # Calendar Features
    # -----------------------------

    def add_datetime_features(self, df):

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

    # -----------------------------
    # Cyclical Features
    # -----------------------------

    def add_cyclical_features(self, df):

        df = df.copy()

        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)

        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)

        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

        df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)

        df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

        return df

    # -----------------------------
    # Lag Features
    # -----------------------------

    def add_lag_features(self, df):

        df = df.copy()

        lags = [1, 2, 3, 6, 12, 24]

        for lag in lags:

            df[f"lag_{lag}"] = df.groupby("Junction")["Vehicles"].shift(lag)

        return df

    # -----------------------------
    # Rolling Statistics
    # -----------------------------

    def add_rolling_features(self, df):

        df = df.copy()

        windows = [3, 6, 12, 24]

        for window in windows:

            rolling = df.groupby("Junction")["Vehicles"].shift(1).rolling(window)

            df[f"rolling_mean_{window}"] = rolling.mean()

        df["rolling_std_24"] = (
            df.groupby("Junction")["Vehicles"].shift(1).rolling(24).std()
        )

        return df

    # -----------------------------
    # Exponential Moving Average
    # -----------------------------

    def add_ema_features(self, df):

        df = df.copy()

        df["ema_12"] = df.groupby("Junction")["Vehicles"].transform(
            lambda x: x.shift(1).ewm(span=12).mean()
        )

        df["ema_24"] = df.groupby("Junction")["Vehicles"].transform(
            lambda x: x.shift(1).ewm(span=24).mean()
        )

        return df

    # -----------------------------
    # Master Function
    # -----------------------------

    def engineer(self, train, test):

        print("=" * 60)
        print("Creating Features")
        print("=" * 60)

        train = self.add_datetime_features(train)
        test = self.add_datetime_features(test)

        train = self.add_cyclical_features(train)
        test = self.add_cyclical_features(test)

        train = self.add_lag_features(train)

        train = self.add_rolling_features(train)

        train = self.add_ema_features(train)

        print("Feature Engineering Complete")

        return train, test

"""
preprocessing.py
----------------
Handles:
1. Loading datasets
2. Data validation
3. Datetime conversion
4. Duplicate removal
5. Missing value report
6. Memory optimization
7. Saving cleaned datasets
"""

from pathlib import Path
import pandas as pd


REQUIRED_TRAIN_COLUMNS = ["DateTime", "Junction", "Vehicles"]
REQUIRED_TEST_COLUMNS = ["DateTime", "Junction"]


class DataPreprocessor:

    def __init__(self, train_path, test_path, processed_dir):

        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.processed_dir = Path(processed_dir)

        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------

    def load_data(self):

        train = pd.read_csv(self.train_path)

        test = pd.read_csv(self.test_path)

        return train, test

    # --------------------------------------------------------

    def validate_columns(self, train, test):

        missing_train = [c for c in REQUIRED_TRAIN_COLUMNS if c not in train.columns]

        missing_test = [c for c in REQUIRED_TEST_COLUMNS if c not in test.columns]

        if missing_train:
            raise ValueError(f"Missing train columns: {missing_train}")

        if missing_test:
            raise ValueError(f"Missing test columns: {missing_test}")

    # --------------------------------------------------------

    def convert_datetime(self, df):

        df = df.copy()

        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

        if df["DateTime"].isna().sum() > 0:
            raise ValueError("Invalid DateTime values found.")

        return df

    # --------------------------------------------------------

    def remove_duplicates(self, df):

        before = len(df)

        df = df.drop_duplicates()

        removed = before - len(df)

        print(f"Removed {removed} duplicate rows.")

        return df

    # --------------------------------------------------------

    def check_missing(self, df, name):

        missing = df.isna().sum()

        print(f"\nMissing Values ({name})")

        print(missing)

        return missing

    # --------------------------------------------------------

    def optimize_memory(self, df):

        if "Junction" in df.columns:
            df["Junction"] = df["Junction"].astype("int8")

        if "Vehicles" in df.columns:
            df["Vehicles"] = df["Vehicles"].astype("float32")

        return df

    # --------------------------------------------------------

    def sort_data(self, df):

        return df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)

    # --------------------------------------------------------

    def save(self, train, test):

        train.to_csv(self.processed_dir / "train_clean.csv", index=False)

        test.to_csv(self.processed_dir / "test_clean.csv", index=False)

    # --------------------------------------------------------

    def preprocess(self):

        print("=" * 60)
        print("Loading Dataset")
        print("=" * 60)

        train, test = self.load_data()

        self.validate_columns(train, test)

        train = self.convert_datetime(train)

        test = self.convert_datetime(test)

        train = self.remove_duplicates(train)

        test = self.remove_duplicates(test)

        self.check_missing(train, "Train")

        self.check_missing(test, "Test")

        train = self.optimize_memory(train)

        test = self.optimize_memory(test)

        train = self.sort_data(train)

        test = self.sort_data(test)

        self.save(train, test)

        print("\nCleaned datasets saved.")

        return train, test

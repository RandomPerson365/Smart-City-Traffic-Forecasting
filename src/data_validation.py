"""Data validation utilities."""
import pandas as pd

def validation_report(df: pd.DataFrame)->pd.DataFrame:
    report=pd.DataFrame({
        "column":df.columns,
        "dtype":[str(t) for t in df.dtypes],
        "missing":df.isna().sum().values,
        "unique":df.nunique().values
    })
    return report

def duplicate_count(df):
    return int(df.duplicated().sum())

import pandas as pd

def descriptive(df):
    return df.describe(include="all")

def correlation(df):
    return df.corr(numeric_only=True)

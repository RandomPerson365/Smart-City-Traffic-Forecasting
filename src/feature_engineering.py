import pandas as pd

def create_features(df):
    df=df.copy()
    df["DateTime"]=pd.to_datetime(df["DateTime"])
    dt=df["DateTime"]
    df["Year"]=dt.dt.year
    df["Month"]=dt.dt.month
    df["Day"]=dt.dt.day
    df["Hour"]=dt.dt.hour
    df["Weekday"]=dt.dt.weekday
    df["Week"]=dt.dt.isocalendar().week.astype(int)
    df["Quarter"]=dt.dt.quarter
    df["DayOfYear"]=dt.dt.dayofyear
    df["IsWeekend"]=(df["Weekday"]>=5).astype(int)
    for lag in [1,2,3,6,12,24,48]:
        df[f"lag_{lag}"]=df.groupby("Junction")["Vehicles"].shift(lag)
    for w in [3,6,12,24]:
        g=df.groupby("Junction")["Vehicles"].shift(1)
        df[f"rolling_mean_{w}"]=g.rolling(w).mean()
        df[f"rolling_std_{w}"]=g.rolling(w).std()
    return df

from pathlib import Path
from feature_engineering import create_features
from data_validation import validation_report
from statistics import descriptive
from visualization import save_distribution
import pandas as pd

def run(train_path,outdir):
    df=pd.read_csv(train_path)
    report=validation_report(df)
    report.to_csv(Path(outdir)/"reports"/"validation_report.csv",index=False)
    df=create_features(df)
    descriptive(df).to_csv(Path(outdir)/"reports"/"descriptive_statistics.csv")
    save_distribution(df,Path(outdir)/"figures"/"traffic_distribution.png")
    df.to_csv(Path(outdir).parent/"data"/"processed"/"train_processed.csv",index=False)

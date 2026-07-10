"""
metrics_report.py

Generate and save model evaluation reports.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd


def save_metrics(results, outfile="outputs/reports/model_metrics.csv"):
    """
    Save model evaluation metrics.

    Parameters
    ----------
    results : list[dict] or DataFrame
        Model evaluation results.

    outfile : str
        Output CSV path.
    """

    output = Path(outfile)

    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    # Save CSV
    df.to_csv(output, index=False)

    # Save readable summary
    summary = output.parent / "metrics_summary.txt"

    with open(summary, "w") as f:

        f.write("=" * 60 + "\n")
        f.write("SMART CITY TRAFFIC FORECASTING\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Generated : {datetime.now()}\n\n")

        if "RMSE" in df.columns:

            best = df.sort_values("RMSE").iloc[0]

            f.write(f"Best Model : {best['Model']}\n\n")

        f.write(df.to_string(index=False))

    print(f"\nMetrics saved to:\n{output}")

    print(f"Summary saved to:\n{summary}")

    return df

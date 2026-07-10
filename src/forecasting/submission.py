from pathlib import Path
import pandas as pd


def build_submission(ids, predictions, outfile="outputs/submissions/submission.csv"):
    """
    Build competition submission file.

    Parameters
    ----------
    ids : iterable
        IDs from test dataset.

    predictions : iterable
        Predicted traffic values.

    outfile : str
        Output CSV path.

    Returns
    -------
    pandas.DataFrame
    """

    output_path = Path(outfile)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"ID": ids, "Vehicles": predictions})

    # Competition expects integer vehicle counts
    df["Vehicles"] = df["Vehicles"].round().clip(lower=0).astype(int)

    df.to_csv(output_path, index=False)

    print(f"\nSubmission saved to:\n{output_path}")

    return df

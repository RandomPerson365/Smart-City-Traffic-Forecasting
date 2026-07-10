import pandas as pd

def build_submission(ids,predictions,outfile="submission.csv"):
    df=pd.DataFrame({
        "ID":ids,
        "Vehicles":predictions
    })
    df.to_csv(outfile,index=False)
    return df

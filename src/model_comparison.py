import pandas as pd

def compare(results,outfile="model_comparison.csv"):
    df=pd.DataFrame(results)
    df.to_csv(outfile,index=False)
    return df

import pandas as pd

def save_metrics(results,outfile="metrics.csv"):
    pd.DataFrame(results).to_csv(outfile,index=False)

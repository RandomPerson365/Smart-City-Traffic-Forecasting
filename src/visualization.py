import matplotlib.pyplot as plt
import seaborn as sns

def save_distribution(df,outfile):
    plt.figure(figsize=(8,4))
    sns.histplot(df["Vehicles"],bins=40,kde=True)
    plt.tight_layout()
    plt.savefig(outfile,dpi=300)
    plt.close()

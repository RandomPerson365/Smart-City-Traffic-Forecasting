import matplotlib.pyplot as plt

def residual_plot(y_true,y_pred,outfile="residuals.png"):
    residual=y_true-y_pred
    plt.figure(figsize=(8,4))
    plt.scatter(y_pred,residual,s=10)
    plt.axhline(0,color="red")
    plt.tight_layout()
    plt.savefig(outfile,dpi=300)
    plt.close()

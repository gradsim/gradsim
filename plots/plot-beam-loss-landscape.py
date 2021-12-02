import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":

    data = None
    curdir = os.path.dirname(os.path.realpath(__file__))
    pkl_path = os.path.join(curdir, "data-beam-loss-landscape.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    kmu_val = data["kmu_val"]
    kmu_loss = data["kmu_loss"]
    klambda_val = data["klambda_val"]
    klambda_loss = data["klambda_loss"]

    sns.set_style("white")
    plt.xlabel("Estimated material parameter value")
    plt.ylabel("(Normalized) Pixelwise MSE between generated video and true video")
    plt.plot(kmu_val, kmu_loss)
    plt.plot(klambda_val, klambda_loss)
    # plt.scatter(1, 0, marker="*", color="r", s=300, linewidths=1)
    plt.legend([
        "First Lame parameter (mu)",
        "Second Lame parameter (lambda)",
    ])
    # savepath = os.path.join(curdir, "plot-beam-loss-landscape.svg")
    # plt.savefig(savepath, format="svg", dpi=1200)
    plt.show()

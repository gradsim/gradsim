import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def normalize(mylist):
    arr = np.asarray(mylist)
    mylist = list((arr - arr.min()) / (arr.max() - arr.min()))
    return mylist


if __name__ == "__main__":

    data = None
    curdir = os.path.dirname(os.path.realpath(__file__))
    pkl_path = os.path.join(curdir, "data-beam-loss-landscape-suppl.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    cmap = ["#94D0FF", "#FF6AD5", "#966BFF", "#FFA58B"]

    kmu_val = data["kmu_val"]
    kmu_loss_1 = normalize(data["kmu_loss_1"])
    klambda_val = data["klambda_val"]
    klambda_loss_1 = normalize(data["klambda_loss_1"])

    sns.set_style("white")
    sns.set_palette(cmap)
    plt.xlabel("Estimated material parameter value")
    plt.ylabel("(Normalized) Pixelwise MSE between generated video and true video")
    plt.plot(kmu_val, kmu_loss_1)
    plt.plot(klambda_val, klambda_loss_1)
    plt.axvline(x=1000.0, linestyle="--", color=cmap[2])
    plt.legend([
        "First lame parameter (mu)",
        "Second lame parameter (lambda)",
        "Ground-truth value",
    ])
    # savepath = os.path.join(curdir, "plot-beam-loss-landscape-suppl.svg")
    # plt.savefig(savepath, format="svg", dpi=1200)
    plt.show()

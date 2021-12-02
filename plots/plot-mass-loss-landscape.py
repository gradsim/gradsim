import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":

    data = None
    curdir = os.path.dirname(os.path.realpath(__file__))
    pkl_path = os.path.join(curdir, "data-mass-loss-landscape.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    masses = data["masses"]

    sns.set_style("white")
    plt.xlabel("Estimated mass")
    plt.ylabel("Pixelwise MSE between generated video and true video")
    plt.plot(masses, data["losses_skip1"])
    plt.plot(masses, data["losses_skip5"], linestyle="--")
    plt.plot(masses, data["losses_first_and_mid"], linestyle="-.")
    plt.plot(masses, data["losses_first_and_last"], linestyle=":")
    plt.scatter(1, 0, marker="*", color="r", s=300, linewidths=1)
    plt.legend([
        "MSE on every frame",
        "MSE on every 5th frame",
        "MSE on first and middle frames",
        "MSE on first and last frames",
        "Ground-truth mass"
    ])
    # savepath = os.path.join(curdir, "plot-mass-loss-landscape.svg")
    # plt.savefig(savepath, format="svg", dpi=1200)
    plt.show()

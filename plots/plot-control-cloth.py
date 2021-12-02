"""
Plot for the `control-cloth` experiment
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import vapeplot


if __name__ == "__main__":

    curdir = os.path.dirname(os.path.realpath(__file__))
    pkl_path = os.path.join(curdir, "data-control-cloth.pkl")
    
    df = pd.read_pickle(pkl_path)

    pal = vapeplot.palette("vaporwave")
    print(pal, len(pal))
    gradsim_color = "#966bff"
    physicsonly_color = "#ff6a8b"
    # noisyphysics_color = "#20de8b"
    random_color = "#94d0ff"

    pal = [gradsim_color, physicsonly_color, random_color]
    sns.lineplot(x="Epoch", y="Position error", hue="Approach", data=df, palette=pal)
    plt.xlabel("Optimization iterations")
    plt.ylabel("Position error (meters)")
    # savepath = os.path.join(curdir, "plot-control-cloth.svg")
    # plt.savefig(savepath, format="svg", dpi=1200)
    plt.show()

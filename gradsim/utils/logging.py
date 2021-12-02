"""
Functions to help logging.
"""

import os

import imageio
import numpy as np


def write_imglist_to_gif(imglist, gifpath, imgformat="rgba", verbose=False):
    """Writes out a list of images into a gif.

    Args:
        imglist (list): List of images (each image is a torch.Tensor).
        gifpath (str): Target filename to save gif to.
        imgformat (str): Format of the input image (organization of dimensions)
            (default: "rgba"; the format that SoftRenderer outupts).
        verbose (bool, Optional): Verbosity flag (default: False).

    Returns:
        (None)
    """
    # List of supported tensor formats
    _valid_formats = ["rgba", "rgb", "dibr"]
    if imgformat not in _valid_formats:
        raise ValueError(f"Got invalid imgformat. Valid formats are {_valid_formats}.")
    writer = imageio.get_writer(gifpath, mode="I")
    for img in imglist:
        if imgformat == "rgba":
            img = img[0].permute(1, 2, 0)
        elif imgformat == "rgb":
            img = img[0]
        elif imgformat == "dibr":
            img = img[0]
        img = img.detach().cpu().numpy()
        writer.append_data((255 * img).astype(np.uint8))
    writer.close()

    if verbose:
        print(f"Saved {gifpath} ({len(imglist)} images).")


def write_imglist_to_dir(imglist, dirname, imgformat="rgba", verbose=False):
    """Writes out a list of images into a directory.

    Args:
        imglist (list): List of images (each image is a torch.Tensor).
        gifpath (str): Target filename to save gif to.
        imgformat (str): Format of the input image (organization of dimensions)
            (default: "rgba"; the format that SoftRenderer outupts).
        verbose (bool, Optional): Verbosity flag (default: False).

    Returns:
        (None)
    """
    # List of supported tensor formats
    _valid_formats = ["rgba", "rgb", "dibr"]
    if imgformat not in _valid_formats:
        raise ValueError(f"Got invalid imgformat. Valid formats are {_valid_formats}.")
    os.makedirs(dirname, exist_ok=True)
    for i, img in enumerate(imglist):
        if imgformat == "rgba":
            img = img[0].permute(1, 2, 0)
        elif imgformat == "rgb":
            img = img[0]
        elif imgformat == "dibr":
            img = img[0]
        img = img.detach().cpu().numpy()
        imageio.imwrite(
            os.path.join(dirname, f"{i:03d}.png"), (255 * img).astype(np.uint8)
        )

    if verbose:
        print(f"Saved {len(imglist)} images to {dirname}.")

# https://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil
import argparse
import colorsys
import glob
import imageio
import numpy as np
import os
import vapeplot

from PIL import Image
from scipy.interpolate import interp1d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory with PNG images to blend")
    parser.add_argument("--out", type=str, default="./blend.png", help="output file name")
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--hue", type=eval, default=True, choices=[True, False])
    parser.add_argument("--transparency", type=eval, default=True, choices=[True, False],
                        help="To apply transparency or not: linear from t_min to t_max from first image to last image")
    parser.add_argument("--t_min", type=float, default=0.5, help="Min transparency value")
    parser.add_argument("--t_max", type=float, default=1.0, help="Max transparency value")
    parser.add_argument("--freq", type=int, default=1)
    parser.add_argument("--r1", type=int, default=None)
    parser.add_argument("--r2", type=int, default=None)
    parser.add_argument("--c1", type=int, default=None)
    parser.add_argument("--c2", type=int, default=None)
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir)
    return args


def hexs(num, t_min=0.5, t_max=1.0):
    hue, sat = get_hue_sat(num)
    ts = np.linspace(t_min, t_max, num)
    return ['#%02x%02x%02x%02x' % (*tuple(int(i*255) for i in c), int(t*255)) for c, t in zip([colorsys.hsv_to_rgb(h, s, 1) for h, s in zip(hue, sat)], ts)]


def get_hue_sat(num):
    # colors = np.array([colorsys.rgb_to_hsv(*tuple(int(c[i:i+2], 16) for i in (1, 3, 5)))[0] for c in vapeplot.palette('vaporwave')])
    # f = interp1d(np.arange(len(colors)), colors)
    # return f(np.arange(num)/(num-1)*(len(colors)-1))
    rgb_start = np.array(hex_to_rgb('#94D0FF'))
    hsv_start = rgb_to_hsv(rgb_start)
    rgb_end = np.array(hex_to_rgb('#FF6AD5'))
    hsv_end = rgb_to_hsv(rgb_end)
    hues = np.linspace(hsv_start[0], hsv_end[0], num) if num > 1 else [hsv_end[0]]
    saturations = np.linspace(hsv_start[1], hsv_end[1], num)  if num > 1 else [hsv_end[1]]
    return hues, saturations


def hex_to_rgb(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv


def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def shift_hue_sat(arr, hout, sout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    hsv[...,1]=sout
    rgb=hsv_to_rgb(hsv)
    return rgb


def blend_images(image_data, out='blend.png', T=None, freq=1,
                 r1=None, r2=None, c1=None, c2=None,
                 hue=True, transparency=True, t_min=0.3, t_max=0.95):

    # image_data can be
    # a directory of PNG files, or
    # a list of 4-D uint8 np images (hxwx4)
    # If dir:
    if isinstance(image_data, str):
        assert os.path.isdir(image_data)
        files = sorted(glob.glob(os.path.join(image_data, '*.png')))[::freq]
        assert len(files), f"No png files found in {image_data}"
        image_data = []
        for f in files:
            im = imageio.imread(f)
            image_data.append(im)
    # Else if list of images:
    else:
        image_data = image_data[::freq]

    num = min(len(image_data), T) if T is not None else len(image_data)
    print(len(image_data), num)

    hues, sats = get_hue_sat(num)
    ts = np.linspace(t_min, t_max, len(image_data)) if num > 1 else [t_max]

    for i in range(num):
        im = image_data[i][r1:r2, c1:c2]
        if hue:
            hue, sat = hues[i], sats[i]
            im = shift_hue_sat(im, hue, sat)
        if transparency:
            im[:, :, -1] = (im[:, :, -1].astype(float)*(ts[i])).astype(np.uint8)
        im = Image.fromarray(im, 'RGBA')
        if i == 0:
            img = im
        else:
            img.paste(im, (0, 0), im)

    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    bg.save(out)


if __name__ == "__main__":

    args = get_args()
    print(args)

    blend_images(args.data_dir, args.out, T=args.T, freq=args.freq,
                 r1=args.r1, r2=args.r2, c1=args.c1, c2=args.c2,
                 hue=args.hue, transparency=args.transparency, t_min=args.t_min, t_max=args.t_max)

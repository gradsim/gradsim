import argparse
import colorsys
import glob
import imageio
import numpy as np
import open3d as o3d    # Import before torch
import os
import time
import torch
import vapeplot

from natsort import natsorted
from PIL import Image
from scipy.interpolate import interp1d
from tqdm import tqdm, trange

from gradsim.utils.logging import write_imglist_to_dir, write_imglist_to_gif
# from image_blend import get_hue_sat, shift_hue_sat


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory with PNG images to blend")
    parser.add_argument("--tmp_dir", type=str, default=None, help="tmp directory (for intermediate imgs)")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory")
    parser.add_argument("--gifpath", type=str, default=None, help="optional gif path to save gif to.")
    parser.add_argument("--rotx", type=float, default=0)
    parser.add_argument("--roty", type=float, default=0)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--palette", type=str, default="vaporwave",
                        choices=["vaporwave", "sunset", "macplus", "jazzcup", "mallsoft", "avanti", "crystal_pepsi", "seapunk", "cool"])
    parser.add_argument("--hue", type=eval, default=True, choices=[True, False])
    parser.add_argument("--transparency", type=eval, default=True, choices=[True, False])
    parser.add_argument("--t_min", type=float, default=0.25)
    parser.add_argument("--t_max", type=float, default=0.95)
    parser.add_argument("--freq", type=int, default=1)
    parser.add_argument("--r1", type=int, default=None)
    parser.add_argument("--r2", type=int, default=None)
    parser.add_argument("--c1", type=int, default=None)
    parser.add_argument("--c2", type=int, default=None)
    parser.add_argument("--tx", type=float, default=0)
    parser.add_argument("--ty", type=float, default=0)
    parser.add_argument("--tz", type=float, default=-5)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    # assert os.path.isdir(args.data_dir)
    return args


def get_colors(palette, num):
    # palette = 'vaporwave'
    # palette = 'cool'
    colors = np.array([colorsys.rgb_to_hsv(*tuple(int(c[i:i+2], 16) for i in (1, 3, 5)))[0] for c in vapeplot.palette(palette)])
    f = interp1d(np.arange(len(colors)), colors)
    return f(np.arange(num)/(num-1)*(len(colors)-1))


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


def shift_hue(arr, hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb


def blend_images_for_suppl(image_data, outdir, freq=1, palette="cool",
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

    colors = get_colors(palette, len(image_data))
    ts = np.linspace(t_min, t_max, len(image_data))

    # num = min(len(image_data), T) if T is not None else len(image_data)
    # print(len(image_data), num)

    # hues, sats = get_hue_sat(num)
    # ts = np.linspace(t_min, t_max, len(image_data)) if num > 1 else [t_max]

    for i in range(len(image_data)):
        im = image_data[i][r1:r2, c1:c2]
        if hue:
            hue = colors[i % len(colors)]
            # hue, sat = hues[i], sats[i]
            im = shift_hue(im, hue)
        if transparency:
            im[:, :, -1] = (im[:, :, -1].astype(float)*(ts[i])).astype(np.uint8)
        im = Image.fromarray(im, 'RGBA')
        # if i == 0:
        #     img = im
        # else:
        #     img.paste(im, (0, 0), im)

        savename = os.path.join(outdir, f"{i:03d}.png")
        im.save(savename)

    # bg = Image.new("RGB", img.size, (255, 255, 255))
    # bg.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    # bg.save(out)


def write_imglist_to_gif_supp(imglist, savename):
    writer = imageio.get_writer(savename, mode="I")
    for img in imglist:
        writer.append_data(img)
    writer.close()


if __name__ == "__main__":

    args = get_args()

    device = "cuda:0"

    gt_files = natsorted(glob.glob(os.path.join(args.data_dir, "[!faces]*.txt")))
    faces = np.loadtxt(os.path.join(args.data_dir, "faces.txt"))
    # faces = torch.from_numpy(faces).long().to(device)
    textures = torch.cat(
        (
            torch.zeros(
                1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
            ),
            torch.zeros(
                1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
            ),
            torch.zeros(
                1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
            ),
        ),
        dim=-1,
    )
    # faces = faces.detach().cpu().numpy()
    f = o3d.utility.Vector3iVector(faces)

    if args.tmp_dir is None:
        args.tmp_dir = os.path.realpath(os.path.join(args.data_dir, "../", os.path.basename(args.data_dir.strip('/'))))
        args.tmp_dir += f"_x{args.tx}y{args.ty}z{args.tz}"
    # if args.out_dir is None:
    #     args.out_dir = os.path.realpath(os.path.join(args.data_dir, "../", os.path.basename(args.data_dir.strip('/'))))
    print("tmp_dir", args.tmp_dir)

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # The last file in each dir is `faces.txt`. Ignore it.
    # for i, gt_file in tqdm(enumerate(gt_files[:args.T]), total=len(gt_files[:args.T])):
    imgs = []
    for i, gt_file in enumerate(gt_files[:args.T]):
        # print(gt_file)
        vertices = np.loadtxt(gt_file)  # .float().to(device)
        # print(vertices.mean())
        v = o3d.utility.Vector3dVector(vertices)
        mesh = o3d.geometry.TriangleMesh(v, f)
        mesh.compute_vertex_normals()
        # mesh.paint_uniform_color([0.5804, 0.8157, 1.0])  # light-blue start color
        mesh.paint_uniform_color([1.0, 0.4157, 0.8352])  # purple, end color
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # vis.get_view_control().rotate(10.0, 10.0)
        vis.add_geometry(mesh)
        mesh.scale(scale=args.scale, center=mesh.get_center())
        # mesh.rotate(
        #     mesh.get_rotation_matrix_from_xyz((np.deg2rad(args.rotx), np.deg2rad(args.roty), 0)),
        #     # center=mesh.get_center(),
        #     center=np.array([0, 0, 0]),
        # )
        # mesh.translate(np.mean(np.asarray(v), axis=0) + np.array([args.tx + 4, args.ty, args.tz]))
        mesh.translate(np.mean(np.asarray(v), axis=0) + np.array([args.tx - 2, args.ty, args.tz]))
        # mesh.translate(np.mean(np.asarray(v), axis=0) + np.array([args.tx, args.ty, args.tz]))
        time.sleep(0.05)
        rgb = np.array(vis.capture_screen_float_buffer(do_render=True))
        # d = np.array(vis.capture_depth_float_buffer(do_render=True))
        rgbd = np.concatenate((rgb, (np.array(vis.capture_depth_float_buffer(do_render=True)) > 0).astype(float)[..., None]), axis=-1)
        img = (rgbd*255).astype(np.uint8)
        img = img[args.r1:args.r2, args.c1:args.c2, :]
        # vis.capture_screen_image(os.path.join(out_gt_dir, f"{i:03d}.png"), do_render=True)
        imageio.imwrite(os.path.join(args.tmp_dir, f"{i:03d}.png"), img)
        imgs.append(img)
        vis.destroy_window()
        # break

    if args.gifpath:
        write_imglist_to_gif_supp(imgs, args.gifpath)

    # # Blend
    # im_name = os.path.realpath(os.path.join(args.tmp_dir, "../", os.path.basename(args.tmp_dir.strip('/')) + f"_f{args.freq}_blend.png"))
    # print("Saving", im_name)
    # blend_images_for_suppl(
    #     args.tmp_dir, args.out_dir, freq=args.freq, palette=args.palette,
    #     r1=args.r1, r2=args.r2, c1=args.c1, c2=args.c2,
    #     hue=args.hue, transparency=args.transparency,
    #     t_min=args.t_min, t_max=args.t_max
    # )

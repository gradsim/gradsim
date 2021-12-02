import argparse
import glob
import imageio
import numpy as np
import open3d as o3d    # Import before torch
import os
import time
import torch
import vapeplot

from natsort import natsorted
from tqdm import tqdm, trange

from gradsim.utils.logging import write_imglist_to_dir, write_imglist_to_gif
from image_blend import blend_images


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory with PNG images to blend")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory")
    parser.add_argument("--rotx", type=float, default=0)
    parser.add_argument("--roty", type=float, default=0)
    parser.add_argument("--rotz", type=float, default=0)
    parser.add_argument("--T", type=int, default=None)
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

    if args.out_dir is None:
        args.out_dir = os.path.realpath(os.path.join(args.data_dir, "../", os.path.basename(args.data_dir.strip('/'))))

    args.out_dir += f"_x{args.tx}y{args.ty}z{args.tz}"
    if args.rotx != 0 or args.roty != 0 or args.rotz != 0:
        args.out_dir += f"_rotx{args.rotx}roty{args.roty}rotz{args.rotz}"
    print("out_dir", args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    # The last file in each dir is `faces.txt`. Ignore it.
    for i, gt_file in tqdm(enumerate(gt_files[:args.T]), total=len(gt_files[:args.T])):
        # print(gt_file)
        vertices = np.loadtxt(gt_file)  # .float().to(device)
        # print(vertices.mean())
        v = o3d.utility.Vector3dVector(vertices)
        mesh = o3d.geometry.TriangleMesh(v, f)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([1, 0.706, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # vis.get_view_control().rotate(10.0, 10.0)
        vis.add_geometry(mesh)
        mesh.scale(scale=args.scale, center=mesh.get_center())
        if args.rotx != 0 or args.roty != 0 or args.rotz != 0:
            mesh.rotate(
                mesh.get_rotation_matrix_from_xyz((np.deg2rad(args.rotx), np.deg2rad(args.roty), np.deg2rad(args.rotz))),
                # center=mesh.get_center(),
                center=np.array([0, 0, 0]),
            )
        mesh.translate(np.mean(np.asarray(v), axis=0) + np.array([args.tx, args.ty, args.tz]))
        rgb = np.array(vis.capture_screen_float_buffer(do_render=True))
        # d = np.array(vis.capture_depth_float_buffer(do_render=True))
        rgbd = np.concatenate((rgb, (np.array(vis.capture_depth_float_buffer(do_render=True)) > 0).astype(float)[..., None]), axis=-1)
        img = (rgbd*255).astype(np.uint8)
        # vis.capture_screen_image(os.path.join(out_gt_dir, f"{i:03d}.png"), do_render=True)
        vis.destroy_window()
        imageio.imwrite(os.path.join(args.out_dir, f"{i:03d}.png"), img)

    # Blend
    im_name = os.path.realpath(os.path.join(args.out_dir, "../", os.path.basename(args.out_dir.strip('/')) + f"_f{args.freq}_blend.png"))
    print("Saving", im_name)
    blend_images(args.out_dir, im_name,
                 freq=args.freq,
                 r1=args.r1, r2=args.r2, c1=args.c1, c2=args.c2,
                 hue=args.hue, transparency=args.transparency, t_min=args.t_min, t_max=args.t_max)

import argparse
import json
import math
import os

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

from argparsers import get_dflex_base_parser
from gradsim import dflex as df
from gradsim.renderutils import SoftRenderer
from gradsim.utils.logging import write_imglist_to_dir, write_imglist_to_gif


def write_meshes_to_file(vertices_across_time, faces, dirname):
    os.makedirs(dirname, exist_ok=True)
    for i, vertices in enumerate(vertices_across_time):
        np.savetxt(os.path.join(dirname, f"{i:03d}.txt"), vertices)
    np.savetxt(os.path.join(dirname, "faces.txt"), faces)


if __name__ == "__main__":

    # Get an argument parser with base-level arguments already filled in.
    dflex_base_parser = get_dflex_base_parser()
    # Create a new parser that inherits these arguments.
    parser = argparse.ArgumentParser(
        parents=[dflex_base_parser], conflict_handler="resolve"
    )

    parser.add_argument(
        "--expid",
        type=str,
        default="default",
        help="Unique string identifier for this experiment.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.join("cache", "control-cloth"),
        help="Directory to store experiment logs in.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=os.path.join("sampledata", "usd", "prop.usda"),
        help="Path to input mesh file (.usda or .tet format).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=os.path.join("sampledata", "target_img_cloth.png"),
        help="Path to target image.",
    )
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=1.5,
        help="Duration of the simulation episode.",
    )
    parser.add_argument(
        "--physics-engine-rate",
        type=int,
        default=60,
        help="Number of physics engine `steps` per 1 second of simulator time.",
    )
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=16,
        help="Number of sub-steps to integrate, per 1 `step` of the simulation.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training iterations."
    )
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")
    parser.add_argument(
        "--method",
        type=str,
        default="gradsim",
        choices=["random", "physics-only", "gradsim"],
        help="Method to use, to optimize for initial velocity."
    )
    parser.add_argument("--log", action="store_true", help="Log experiment data.")

    args = parser.parse_args()
    print(args)

    # torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(args.seed)

    torch.autograd.set_detect_anomaly(True)

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
    sim_steps = int(args.sim_duration / sim_dt)
    sim_time = 0.0

    # Sample a random height in the range [2, 3) meters.
    # height = 2.5
    height = (2.0 - 3.0) * torch.rand(1) + 3.0

    clothdims = torch.randint(low=8, high=32, size=(1,))

    print(f"Cloth dims: {clothdims}")

    initial_vel = (0.0, 0.0, 0.0)
    if args.method == "random":
        initial_vel = (1.5, 0.0, 0.0)

    builder = df.sim.ModelBuilder()
    builder.add_cloth_grid(
        pos=(-5.0, height, 0.0),
        rot=df.quat_from_axis_angle((1.0, 0.5, 0.0), math.pi * 0.5),
        vel=initial_vel,
        dim_x=clothdims,
        dim_y=clothdims,
        cell_x=0.125,
        cell_y=0.125,
        mass=2.0,
    )  # , fix_left=True, fix_right=True, fix_top=True, fix_bottom=True)

    model = builder.finalize("cpu")
    model.tri_lambda = 10000.0
    model.tri_ka = 10000.0
    model.tri_kd = 100.0
    model.tri_lift = 10.0
    model.tri_drag = 5.0

    model.contact_ke = 1.0e4
    model.contact_kd = 1000.0
    model.contact_kf = 1000.0
    model.contact_mu = 0.5

    model.particle_radius = 0.01
    model.ground = False

    target = torch.tensor((3.5, 0.0, 0.0))
    initial_velocity = torch.tensor((1.0, 0.0, 0.0), requires_grad=True)

    integrator = df.sim.SemiImplicitIntegrator()

    # Setup SoftRasterizer
    device = "cuda:0"
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 15.0
    elevation = 0.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    faces = model.tri_indices
    textures = torch.cat(
        (
            torch.zeros(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
            torch.ones(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )

    # target_image = imageio.imread("cache/control-cloth/debug/target_img_end.png")
    # target_image = torch.from_numpy(target_image).float().to(device) / 255.0
    # target_image = target_image.permute(2, 0, 1).unsqueeze(0)

    target_position = torch.zeros(3, dtype=torch.float32)
    target_position[0] = (0.0 - 3.0) * torch.rand(1) + 3.0
    target_position[1] = -3.0
    target_image = None
    with torch.no_grad():
        builder_gt = df.sim.ModelBuilder()
        builder_gt.add_cloth_grid(
            pos=(target_position[0].item(), target_position[1].item(), target_position[2].item()),
            rot=df.quat_from_axis_angle((1.0, 0.5, 0.0), math.pi * 0.5),
            vel=initial_vel,
            dim_x=clothdims,
            dim_y=clothdims,
            cell_x=0.125,
            cell_y=0.125,
            mass=1.0,
        )  # , fix_left=True, fix_right=True, fix_top=True, fix_bottom=True)

        model_gt = builder_gt.finalize("cpu")
        model_gt.tri_lambda = 10000.0
        model_gt.tri_ka = 10000.0
        model_gt.tri_kd = 100.0
        model_gt.tri_lift = 10.0
        model_gt.tri_drag = 5.0

        model_gt.contact_ke = 1.0e4
        model_gt.contact_kd = 1000.0
        model_gt.contact_kf = 1000.0
        model_gt.contact_mu = 0.5

        model_gt.particle_radius = 0.01
        model_gt.ground = False
        state_gt = model_gt.state()
        device = "cuda:0"
        target_image = renderer.forward(
            state_gt.q.unsqueeze(0).to(device),
            faces.unsqueeze(0).to(device),
            textures.to(device),
        )
        if args.log:
            imageio.imwrite(
                os.path.join(logdir, "target_image.png"),
                (target_image[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
                    np.uint8
                ),
            )
            np.savetxt(
                os.path.join(logdir, "target_position.txt"),
                target_position.detach().cpu().numpy(),
            )
            os.makedirs(os.path.join(logdir, "gt"), exist_ok=True)
            np.savetxt(
                os.path.join(logdir, "gt", "vertices_gt.txt"),
                state_gt.q.detach().cpu().numpy(),
            )
            np.savetxt(
                os.path.join(logdir, "gt", "faces.txt"),
                faces.detach().cpu().numpy(),
            )

    param = initial_velocity

    # optimizer = torch.optim.SGD([param], lr=1e-3, momentum=0.4)
    optimizer = torch.optim.Adam([param], lr=args.lr)

    render_every = 60

    losses = []
    position_errors = []
    initial_velocities = []

    try:

        for e in trange(args.epochs):

            # reset state
            sim_time = 0.0
            state = model.state()

            state.u = state.u + initial_velocity
            if args.method == "random":
                state.u = state.u + torch.rand_like(state.u)

            initial_velocities.append(list(initial_velocity.detach().cpu().numpy()))

            loss = torch.zeros(1, requires_grad=True)
            if args.method == "gradsim":
                loss = loss.to(device)

            imgs = []
            vertices = []

            # run simulation
            for i in range(0, sim_steps):

                state = integrator.forward(model, state, sim_dt)

                if i % render_every == 0 or i == sim_steps - 1:
                    # with torch.no_grad():
                    device = "cuda:0"
                    rgba = renderer.forward(
                        state.q.unsqueeze(0).to(device),
                        faces.unsqueeze(0).to(device),
                        textures.to(device),
                    )
                    imgs.append(rgba)
                    vertices.append(state.q.detach().cpu().numpy())

                sim_time += sim_dt

                # compute loss
                com_pos = torch.mean(state.q, 0) + 1e-16
                com_vel = torch.mean(state.u, 0)

                # use integral of velocity over course of the run
                if i % render_every == 0 or i == sim_steps - 1:
                    # To ensure fairness: both methods get loss at same timesteps.
                    if args.method == "physics-only":
                        loss = loss + torch.nn.functional.mse_loss(com_pos, target_position)
                    if args.method == "gradsim":
                        loss = loss + torch.nn.functional.mse_loss(
                            imgs[-1] + 1e-16, target_image
                        ) + 1e-5

            # print("method:", args.method, initial_velocity)
            # if args.method == "gradsim":
            #     loss = (
            #         torch.nn.functional.mse_loss(imgs[-1], target_image)
            #         + torch.nn.functional.mse_loss(imgs[-2], target_image)
            #         + torch.nn.functional.mse_loss(imgs[-3], target_image)
            #         + torch.nn.functional.mse_loss(imgs[-4], target_image)
            #         + torch.nn.functional.mse_loss(imgs[-5], target_image)
            #     )

            if args.log:
                write_imglist_to_gif(
                    imgs,
                    os.path.join(logdir, f"{e:02d}.gif"),
                    imgformat="rgba",
                    verbose=False,
                )
                write_imglist_to_dir(
                    imgs, os.path.join(logdir, f"{e:02d}"), imgformat="rgba",
                )
                imageio.imwrite(
                    os.path.join(logdir, f"last_frame_{e:02d}.png"),
                    (imgs[-1][0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
                        np.uint8
                    ),
                )
                write_meshes_to_file(
                    vertices,
                    faces.detach().cpu().numpy(),
                    os.path.join(logdir, f"vertices_{e:05d}")
                )

            if args.method != "random":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if e == 4:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = param_group["lr"] / 10
                # if e == 30:
                #     for pg in optimizer.param_groups:
                #         param_group["lr"] = param_group["lr"] / 2

            position_error = torch.nn.functional.mse_loss(
                state.q.mean(0), target_position
            )

            tqdm.write(f"Loss: {loss.item():.5}")
            tqdm.write(f"Position error: {position_error.item():.5}")

            losses.append(loss.item())
            position_errors.append(position_error.item())

    except Exception as e:
        with open(os.path.join(logdir, "exceptions.txt"), "w") as f:
            f.write("Exception occured!\n")
        f.close()

    if args.log:
        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "position_errors.txt"), position_errors)
        np.savetxt(os.path.join(logdir, "initial_velocities.txt"), np.array(initial_velocities))

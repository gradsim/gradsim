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
from pxr import Usd, UsdGeom


def write_meshes_to_file(vertices_across_time, faces, dirname):
    os.makedirs(dirname, exist_ok=True)
    for i, vertices in enumerate(vertices_across_time):
        np.savetxt(os.path.join(dirname, f"{i:03d}.txt"), vertices)
    np.savetxt(os.path.join(dirname, "faces.txt"), faces)


def read_tet_mesh(filepath):
    vertices = []
    faces = []
    with open(filepath, "r") as mesh:
        for line in mesh:
            data = line.split()
            if len(data) == 0:
                continue
            if data[0] == "v":
                vertices.append([float(d) for d in data[1:]])
            elif data[0] == "t":
                faces.append([int(d) for d in data[1:]])
    # vertices = torch.tensor([float(el) for sublist in vertices for el in sublist]).view(-1, 3)
    # faces = torch.tensor(faces, dtype=torch.long)
    vertices = [tuple(v) for v in vertices]  # Does not seem to work
    vertices = np.asarray(vertices).astype(np.float32)
    faces = [f for face in faces for f in face]
    return vertices, faces


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
        default=os.path.join("cache", "control-fem"),
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
        default=os.path.join("sampledata", "target_img_fem_gear.png"),
        help="Path to target image.",
    )
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=3.0,
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
        default=32,
        help="Number of sub-steps to integrate, per 1 `step` of the simulation.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training iterations."
    )
    parser.add_argument("--lr", type=float, default=10, help="Learning rate.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "physics-only", "noisy-physics-only", "gradsim"],
    )
    parser.add_argument("--log", action="store_true", help="Log experiment data.")

    args = parser.parse_args()
    print(args)

    torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(args.seed)

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
    sim_steps = int(args.sim_duration / sim_dt)
    sim_time = 0.0

    phase_count = 8
    phase_step = math.pi / phase_count * 2.0
    phase_freq = 2.5

    builder = df.sim.ModelBuilder()

    if args.mesh[-4:] == "usda":

        mesh = Usd.Stage.Open(args.mesh)
        geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/mesh"))
        points = geom.GetPointsAttr().Get()
        tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()
        # tri_indices = geom.GetFaceVertexIndicesAttr().Get()
        # tri_counts = geom.GetFaceVertexCountsAttr().Get()

    elif args.mesh[-3:] == "tet":
        points, tet_indices = read_tet_mesh(args.mesh)

    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.0),
    )

    builder.add_soft_mesh(
        pos=(-4.0, 2.0, 0.0),
        rot=r,
        scale=1.0,
        vel=(1.5, 0.0, 0.0),
        vertices=points,
        indices=tet_indices,
        density=1.0,
    )

    model = builder.finalize("cpu")

    model.tet_kl = 1000.0
    model.tet_km = 1000.0
    model.tet_kd = 1.0

    model.tri_ke = 0.0
    model.tri_ka = 0.0
    model.tri_kd = 0.0
    model.tri_kb = 0.0

    model.contact_ke = 1.0e4
    model.contact_kd = 1.0
    model.contact_kf = 10.0
    model.contact_mu = 0.5

    model.particle_radius = 0.05
    model.ground = True
    # model.gravity = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32, requires_grad=False)

    rest_angle = model.edge_rest_angle

    # one fully connected layer + tanh activation
    network = torch.nn.Sequential(
        torch.nn.Linear(phase_count, model.tet_count, bias=False), torch.nn.Tanh()
    )

    activation_strength = 0.3
    activation_penalty = 0.0

    render_time = 0

    integrator = df.sim.SemiImplicitIntegrator()

    # Setup SoftRasterizer
    device = "cuda:0"
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 13.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    faces = model.tri_indices
    textures = torch.cat(
        (
            torch.ones(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
            torch.ones(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )

    # target_image = imageio.imread(args.target)
    # target_image = torch.from_numpy(target_image).float().to(device) / 255.0
    # target_image = target_image.permute(2, 0, 1).unsqueeze(0)

    target_position = torch.zeros(3, dtype=torch.float32)
    target_position[0] = 4.0
    target_position[1] = 2.0
    target_image = None
    print("Target position:", target_position)
    with torch.no_grad():
        gt_builder = df.sim.ModelBuilder()
        gt_builder.add_soft_mesh(
            pos=(target_position[0], target_position[1], target_position[2]),
            rot=r,
            scale=1.0,
            vel=(1.5, 0.0, 0.0),
            vertices=points,
            indices=tet_indices,
            density=1.0,
        )
        gt_model = gt_builder.finalize("cpu")
        gt_model.tet_kl = model.tet_kl
        gt_model.tet_km = model.tet_km
        gt_model.tet_kd = model.tet_kd
        gt_model.tri_ke = model.tri_ke
        gt_model.tri_ka = model.tri_ka
        gt_model.tri_kd = model.tri_kd
        gt_model.tri_kb = model.tri_kb
        gt_model.contact_ke = model.contact_ke
        gt_model.contact_kd = model.contact_kd
        gt_model.contact_kf = model.contact_kf
        gt_model.contact_mu = model.contact_mu
        gt_model.particle_radius = model.particle_radius
        gt_model.ground = model.ground
        gt_state = gt_model.state()
        device = "cuda:0"
        target_image = renderer.forward(
            gt_state.q.unsqueeze(0).to(device),
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
            np.savetxt(
                os.path.join(logdir, "vertices_gt.txt"),
                gt_state.q.detach().cpu().numpy()
            )
            np.savetxt(
                os.path.join(logdir, "faces_gt.txt"),
                faces.detach().cpu().numpy()
            )

    # optimizer = torch.optim.SGD(network.parameters(), lr=train_rate, momentum=0.5)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    losses = []
    position_errors = []

    try:

        for e in range(args.epochs):

            sim_time = 0.0

            state = model.state()

            loss = torch.zeros(1, requires_grad=True)

            print_every = 60 * 32
            render_every = 60 * 8

            imgs = []
            vertices = []

            for i in trange(0, sim_steps):

                phases = torch.zeros(phase_count)
                for p in range(phase_count):
                    phases[p] = math.sin(phase_freq * (sim_time + p * phase_step))
                # compute activations (rest angles)
                if args.method == "random":
                    model.tet_activations = (
                        (-1.0 - 1.0) * torch.rand(model.tet_count) + 1
                    ) * activation_strength
                else:
                    model.tet_activations = network(phases) * activation_strength

                state = integrator.forward(model, state, sim_dt)
                sim_time += sim_dt

                if i % render_every == 0:
                    # with torch.no_grad():
                    device = "cuda:0"
                    rgba = renderer.forward(
                        state.q.unsqueeze(0).to(device),
                        faces.unsqueeze(0).to(device),
                        textures.to(device),
                    )
                    imgs.append(rgba)
                    vertices.append(state.q.detach().cpu().numpy())

            loss = torch.nn.functional.mse_loss(imgs[-1], target_image)
            position_error = torch.nn.functional.mse_loss(
                state.q.mean(0), target_position
            )

            tqdm.write(f"Loss: {loss.item():.5}")
            tqdm.write(f"Position error: {position_error.item():.5}")

            losses.append(loss.item())
            position_errors.append(position_error.item())

            if args.method == "random":
                pass  # The random exploration baseline does not need backprop
            else:
                if args.method == "noisy-physics-only":
                    noisy_position_error = torch.nn.functional.mse_loss(
                        state.q.mean(0),
                        target_position
                        + (torch.rand_like(target_position) * 0.1 * target_position),
                    )
                    noisy_position_error.backward()
                elif args.method == "physics-only":
                    position_error.backward()
                elif args.method == "gradsim":
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()

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

    except KeyboardInterrupt:
        pass

    if args.log:
        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "position_errors.txt"), position_errors)

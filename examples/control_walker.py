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
        default=os.path.join("cache", "control-walker"),
        help="Directory to store experiment logs in.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=os.path.join("sampledata", "usd", "walker.usda"),
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
        default=5.0,
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
        "--epochs", type=int, default=40, help="Number of training iterations."
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
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

    # sim_duration = 5.0  # seconds
    # sim_substeps = 32
    # sim_dt = (1.0 / 60.0) / sim_substeps
    # sim_steps = int(sim_duration / sim_dt)
    # sim_time = 0.0

    train_rate = 0.001  # 0.0001

    phase_count = 4

    builder = df.sim.ModelBuilder()

    walker = Usd.Stage.Open(args.mesh)
    mesh = UsdGeom.Mesh(walker.GetPrimAtPath("/Grid/Grid"))

    points = mesh.GetPointsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()

    for p in points:
        builder.add_particle(tuple(p), (0.0, 0.0, 0.0), 1.0)

    for t in range(0, len(indices), 3):
        i = indices[t + 0]
        j = indices[t + 1]
        k = indices[t + 2]

        builder.add_triangle(i, j, k)

    model = builder.finalize("cpu")
    # model.tri_lambda = 10000.0
    # model.tri_ka = 10000.0
    # model.tri_kd = 100.0

    model.tri_ke = 10000.0
    model.tri_ka = 10000.0
    model.tri_kd = 100.0
    model.tri_lift = 0.0
    model.tri_drag = 0.0

    edge_ke = 0.0
    edge_kd = 0.0

    model.contact_ke = 1.0e4
    model.contact_kd = 1000.0
    model.contact_kf = 1000.0
    model.contact_mu = 0.5

    model.particle_radius = 0.01

    # one fully connected layer + tanh activation
    network = torch.nn.Sequential(
        torch.nn.Linear(phase_count, model.tri_count, bias=False), torch.nn.Tanh()
    )

    activation_strength = 0.2
    activation_penalty = 0.1

    integrator = df.sim.SemiImplicitIntegrator()

    # Setup SoftRasterizer
    device = "cuda:0"
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 8.0
    elevation = 30.0
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

    target_image = imageio.imread(os.path.join("sampledata", "target_img_walker.png"))
    target_image = torch.from_numpy(target_image).float().to(device) / 255.0
    target_image = target_image.permute(2, 0, 1).unsqueeze(0)

    # optimizer = torch.optim.SGD(network.parameters(), lr=train_rate, momentum=0.25)
    # TODO: Tune the learning rate (the current one is likely too low)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    # epochs = 40
    render_every = 60 * 4

    losses = []
    position_xs = []

    for e in trange(args.epochs):

        sim_time = 0.0

        state = model.state()

        loss = torch.zeros(1, requires_grad=True)

        imgs = []

        for i in range(0, sim_steps):

            # build sinusoidal phase inputs
            phases = torch.zeros(phase_count)
            for p in range(phase_count):
                phases[p] = math.sin(20.0 * (sim_time + 0.5 * p * math.pi))

            model.tri_activations = network(phases) * activation_strength
            state = integrator.forward(model, state, sim_dt)

            sim_time += sim_dt

            # if (render and (i%sim_substeps == 0)):
            #     render_time += sim_dt*sim_substeps
            # renderer.update(state, render_time)
            if i % render_every == 0 or i == sim_steps - 1:
                # with torch.no_grad():
                device = "cuda:0"
                rgba = renderer.forward(
                    state.q.unsqueeze(0).to(device),
                    faces.unsqueeze(0).to(device),
                    textures.to(device),
                )
                imgs.append(rgba)

            com_pos = torch.mean(state.q, 0)
            com_vel = torch.mean(state.u, 0)

            """
            TODO: Apart from the model.tri_activation variable, no other
            term (not even state.q.mean(0) or state.com.mean(0)) seems to
            affect the loss function. Look into this.
            """

            if args.method == "physics-only":
                # use integral of velocity over course of the run
                loss = (
                    loss
                    - com_vel[0]
                    + torch.norm(model.tri_activations) * activation_penalty
                )

        if args.method == "gradsim":
            loss = torch.nn.functional.mse_loss(imgs[-1], target_image)
        tqdm.write(f"Loss: {loss.item():.5}")

        # com_pos = state.q.mean(0)
        # com_pos_err = (com_pos - torch.tensor([4.0, 0.0, 0.0])) ** 2
        # loss = loss - state.u.mean(0)[0]

        losses.append(loss.item())
        position_xs.append(state.q.mean(0)[0].item())

        if args.method != "random":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        tqdm.write(f"Iter: {e:03d}, Loss: {loss.item():.5}")

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

    if args.log:
        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "position_xs.txt"), position_xs)

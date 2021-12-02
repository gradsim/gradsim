import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import trange

from argparsers import get_dflex_base_parser
from gradsim import dflex as df
from gradsim.renderutils import SoftRenderer
from gradsim.utils.logging import write_imglist_to_dir, write_imglist_to_gif


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


class SimpleModel(torch.nn.Module):
    """A thin wrapper around a parameter, for convenient use with optim. """

    def __init__(self, param, activation=None):
        super(SimpleModel, self).__init__()
        self.update = torch.nn.Parameter(torch.rand(param.shape) * 0.1)
        self.param = param
        self.activation = activation

    def forward(self):
        out = self.param + self.update
        if self.activation is not None:
            return self.activation(out) + 1e-8
        return out


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
        default=os.path.join("cache", "demo-fem"),
        help="Directory to store experiment logs in.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=os.path.join("sampledata", "tet", "icosphere.tet"),
        help="Path to input mesh file (.tet format).",
    )
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=2.0,
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
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")
    parser.add_argument(
        "--method",
        type=str,
        default="gradsim",
        choices=["noisy-physics-only", "physics-only", "gradsim"],
        help="Method to use, to optimize for initial velocity."
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=1,
        help="Interval at which video frames are compared.",
    )
    parser.add_argument("--log", action="store_true", help="Log experiment data.")

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    torch.autograd.set_detect_anomaly(True)

    device = "cuda:0"

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
    sim_steps = int(args.sim_duration / sim_dt)
    sim_time = 0.0

    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 8.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    render_steps = args.sim_substeps

    phase_count = 8
    phase_step = math.pi / phase_count * 2.0
    phase_freq = 2.5

    # points, tet_indices = read_tet_mesh("cache/tetassets/icosphere.tet")
    points, tet_indices = read_tet_mesh(args.mesh)
    # print(points)
    # print(tet_indices)

    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.0),
    )

    vx_init = (1.0 - 3.0) * torch.rand(1) + 3.0
    # pos = torch.tensor([0.0, 2.0, 0.0])
    # vel = torch.tensor([1.5, 0.0, 0.0])

    imgs_gt = []

    particle_inv_mass_gt = None

    with torch.no_grad():
        builder_gt = df.sim.ModelBuilder()
        builder_gt.add_soft_mesh(
            pos=(-2.0, 2.0, 0.0),
            rot=r,
            scale=1.0,
            vel=(vx_init.item(), 0.0, 0.0),
            vertices=points,
            indices=tet_indices,
            density=10.0,
        )

        model_gt = builder_gt.finalize("cpu")

        model_gt.tet_kl = 1000.0
        model_gt.tet_km = 1000.0
        model_gt.tet_kd = 1.0

        model_gt.tri_ke = 0.0
        model_gt.tri_ka = 0.0
        model_gt.tri_kd = 0.0
        model_gt.tri_kb = 0.0

        model_gt.contact_ke = 1.0e4
        model_gt.contact_kd = 1.0
        model_gt.contact_kf = 10.0
        model_gt.contact_mu = 0.5

        model_gt.particle_radius = 0.05
        model_gt.ground = True
        # model_gt.gravity = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32, requires_grad=False)

        particle_inv_mass_gt = model_gt.particle_inv_mass.clone()

        rest_angle = model_gt.edge_rest_angle

        # one fully connected layer + tanh activation
        network = torch.nn.Sequential(
            torch.nn.Linear(phase_count, model_gt.tet_count, bias=False),
            torch.nn.Tanh(),
        )

        activation_strength = 0.3
        activation_penalty = 0.0

        integrator = df.sim.SemiImplicitIntegrator()

        sim_time = 0.0

        state_gt = model_gt.state()
        # loss = torch.zeros(1, requires_grad=True)

        faces = model_gt.tri_indices
        textures = torch.cat(
            (
                torch.ones(
                    1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                ),
                torch.ones(
                    1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                ),
                torch.zeros(
                    1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                ),
            ),
            dim=-1,
        )

        # states_gt = []
        imgs_gt = []
        positions_gt = []
        for i in trange(0, sim_steps):
            # phases = torch.zeros(phase_count)
            # for p in range(phase_count):
            #     phases[p] = math.sin(phase_freq * (sim_time + p * phase_step))
            # # compute activations (rest angles)
            # model_gt.tet_activations = network(phases) * activation_strength

            state_gt = integrator.forward(model_gt, state_gt, sim_dt)
            sim_time += sim_dt

            if i % render_steps == 0:
                rgba = renderer.forward(
                    state_gt.q.unsqueeze(0).to(device),
                    faces.unsqueeze(0).to(device),
                    textures.to(device),
                )
                imgs_gt.append(rgba)
                positions_gt.append(state_gt.q)
        if args.log:
            write_imglist_to_gif(
                imgs_gt,
                os.path.join(logdir, "gt.gif"),
                imgformat="rgba",
                verbose=False,
            )
            write_imglist_to_dir(
                imgs_gt, os.path.join(logdir, "gt"), imgformat="rgba",
            )
            # write_imglist_to_gif(
            #     imgs_gt, "cache/fem/gt.gif", imgformat="rgba", verbose=False
            # )
            np.savetxt(
                os.path.join(logdir, "mass_gt.txt"),
                particle_inv_mass_gt.detach().cpu().numpy(),
            )
            np.savetxt(
                os.path.join(logdir, "vertices.txt"),
                state_gt.q.detach().cpu().numpy()
            )
            np.savetxt(
                os.path.join(logdir, "face.txt"),
                faces.detach().cpu().numpy()
            )

    massmodel = SimpleModel(
        # particle_inv_mass_gt + 50 * torch.rand_like(particle_inv_mass_gt),
        particle_inv_mass_gt + 0.1 * torch.rand_like(particle_inv_mass_gt),
        activation=torch.nn.functional.relu,
    )
    # epochs = 100
    save_gif_every = 1
    # compare_every = 10

    optimizer = torch.optim.Adam(massmodel.parameters(), lr=1e-1)
    # optimizer = torch.optim.LBFGS(massmodel.parameters(), lr=1.0, tolerance_grad=1.e-5, tolerance_change=0.01, line_search_fn ="strong_wolfe")
    lossfn = torch.nn.MSELoss()

    try:

        for e in range(args.epochs):

            # if e in [20, 40]:
            #     for param_group in optimizer.param_groups:
            #         param_group["lr"] = param_group["lr"] * 0.1

            builder = df.sim.ModelBuilder()
            builder.add_soft_mesh(
                pos=(-2.0, 2.0, 0.0),
                rot=r,
                scale=1.0,
                vel=(vx_init.item(), 0.0, 0.0),
                vertices=points,
                indices=tet_indices,
                density=10.0,
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

            model.particle_inv_mass = massmodel()
            # print(torch.allclose(massmodel(), particle_mass_gt))

            rest_angle = model.edge_rest_angle

            # one fully connected layer + tanh activation
            network = torch.nn.Sequential(
                torch.nn.Linear(phase_count, model.tet_count, bias=False), torch.nn.Tanh()
            )

            activation_strength = 0.3
            activation_penalty = 0.0

            integrator = df.sim.SemiImplicitIntegrator()

            sim_time = 0.0

            state = model.state()
            # loss = torch.zeros(1, requires_grad=True)

            faces = model.tri_indices
            textures = torch.cat(
                (
                    torch.ones(
                        1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                    ),
                    torch.ones(
                        1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                    ),
                    torch.zeros(
                        1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                    ),
                ),
                dim=-1,
            )

            # states = []
            imgs = []
            positions = []
            losses = []
            inv_mass_errors = []
            mass_errors = []
            for i in trange(0, sim_steps):
                # phases = torch.zeros(phase_count)
                # for p in range(phase_count):
                #     phases[p] = math.sin(phase_freq * (sim_time + p * phase_step))
                # # compute activations (rest angles)
                # model.tet_activations = network(phases) * activation_strength

                state = integrator.forward(model, state, sim_dt)
                sim_time += sim_dt

                if i % render_steps == 0:
                    rgba = renderer.forward(
                        state.q.unsqueeze(0).to(device),
                        faces.unsqueeze(0).to(device),
                        textures.to(device),
                    )
                    imgs.append(rgba)
                    positions.append(state.q)

            if args.method == "gradsim":
                loss = sum(
                    [
                        lossfn(est, gt)
                        for est, gt in zip(
                            imgs[::args.compare_every],
                            imgs_gt[::args.compare_every]
                        )
                    ]
                ) / len(imgs[::args.compare_every])
            elif args.method == "physics-only":
                loss = sum(
                    [
                        lossfn(est, gt)
                        for est, gt in zip(
                            positions[::args.compare_every],
                            positions_gt[::args.compare_every]
                        )
                    ]
                )
            elif args.method == "noisy-physics-only":
                loss = sum(
                    [
                        lossfn(est, gt + torch.rand_like(gt) * 0.1)
                        for est, gt in zip(
                            positions[::args.compare_every],
                            positions_gt[::args.compare_every],
                        )
                    ]
                )
            inv_mass_err = lossfn(model.particle_inv_mass, particle_inv_mass_gt)
            mass_err = lossfn(
                1 / (model.particle_inv_mass + 1e-6), 1 / (particle_inv_mass_gt + 1e-6)
            )
            print(
                f"[EPOCH: {e:03d}] "
                f"Loss: {loss.item():.5f} (Inv) Mass err: {inv_mass_err.item():.5f} "
                f"Mass err: {mass_err.item():.5f}"
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            inv_mass_errors.append(inv_mass_err.item())
            mass_errors.append(mass_err.item())

            if args.log and ((e % save_gif_every == 0) or (e == epochs - 1)):
                write_imglist_to_gif(
                    imgs, os.path.join(logdir, f"{e:05d}.gif"), imgformat="rgba"
                )
                write_imglist_to_dir(
                    imgs, os.path.join(logdir, f"{e:05d}"), imgformat="rgba"
                )
                np.savetxt(
                    os.path.join(logdir, f"mass_{e:05d}.txt"),
                    model.particle_inv_mass.detach().cpu().numpy(),
                )

    except KeyboardInterrupt:
        pass

    if args.log:
        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "inv_mass_errors.txt"), inv_mass_errors)
        np.savetxt(os.path.join(logdir, "mass_errors.txt"), mass_errors)

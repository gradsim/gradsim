"""
Optimize parameters of a simple pendulum from video.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm, trange

from gradsim.bodies import SimplePendulum
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.utils import meshutils
from gradsim.utils.logging import write_imglist_to_gif


class SimpleModel(torch.nn.Module):
    """A thin wrapper around a parameter, for convenient use with optim. """

    def __init__(self, param):
        super(SimpleModel, self).__init__()
        self.update = torch.nn.Parameter(torch.rand(param.shape) * 0.001)
        self.param = param

    def forward(self):
        return self.param + self.update


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expid",
        type=str,
        default="default",
        help="Unique string identifier for experiments.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=Path("cache/simple_pendulum"),
        help="Directory to store logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (for repeatability)"
    )
    parser.add_argument(
        "--starttime", type=float, default=0.0, help="Simulation start time (sec)"
    )
    parser.add_argument(
        "--simsteps",
        type=int,
        default=50,
        help="Number of timesteps to run simulation for",
    )
    parser.add_argument(
        "--dtime", type=float, default=0.1, help="Simulation timestep size (sec)"
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.91,
        help="Acceleration due to gravity (m/s^2)",
    )
    parser.add_argument(
        "--length", type=float, default=1.0, help="Length of the pendulum (m)"
    )
    parser.add_argument(
        "--damping", type=float, default=0.5, help="Damping coefficient"
    )
    parser.add_argument("--mass", type=float, default=1.0, help="Mass of the bob (kg)")
    parser.add_argument(
        "--radius", type=float, default=1.0, help="Radius of the bob (m)"
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=10,
        help="Apply loss every `--compare-every` frames.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to run parameter optimization for",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=Path("sampledata/sphere.obj"),
        help="Path to template sphere mesh (.obj) file.",
    )
    parser.add_argument(
        "--optimize-length",
        action="store_true",
        help="Optimize for the length of the pendulum.",
    )
    parser.add_argument(
        "--optimize-gravity",
        action="store_true",
        help="Optimize for the acceleration due to gravity.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")

    args = parser.parse_args()

    if args.compare_every >= args.simsteps:
        raise ValueError(
            f"Arg --compare-every cannot be greater than or equal to {args.simsteps}."
        )

    # Seed RNG for repeatability
    torch.manual_seed(args.seed)

    # Device the tensors are all stored on.
    device = "cuda:0"

    # Load the template sphere mesh (for the bob)
    sphere = TriangleMesh.from_obj(args.template)
    vertices_gt = meshutils.normalize_vertices(
        sphere.vertices.unsqueeze(0), scale_factor=args.radius
    ).to(device)
    faces_gt = sphere.faces.to(device).unsqueeze(0)
    textures = torch.cat(
        (
            torch.ones(1, faces_gt.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces_gt.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces_gt.shape[1], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )

    # Initialize the renderer.
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 8.0
    elevation = 0.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    pendulum_gt = SimplePendulum(
        args.mass, args.radius, args.gravity, args.length, args.damping
    ).to(device)
    theta_init = torch.tensor([0.0, 3.0], device=device)
    times = torch.arange(
        args.starttime,
        args.starttime + args.simsteps * args.dtime,
        args.dtime,
        device=device,
    )
    numsteps = times.numel()
    ret = odeint(pendulum_gt, theta_init, times)

    theta1_gt = ret[:, 0]
    theta2_gt = ret[:, 1]

    # Simulation

    x_gt = args.length * theta1_gt.sin()
    y_gt = -args.length * theta1_gt.cos()
    z_gt = torch.zeros_like(x_gt)
    pos_gt = torch.stack((x_gt, y_gt, z_gt), dim=-1)

    imgs_gt = []
    print("Rendering GT images...")
    for i in trange(numsteps):
        _vertices = vertices_gt.clone() + pos_gt[i]
        rgba = renderer.forward(_vertices, faces_gt, textures)
        imgs_gt.append(rgba)

    logdir = Path(args.logdir) / args.expid
    if args.log:
        logdir.mkdir(exist_ok=True)

    if args.log:
        write_imglist_to_gif(
            imgs_gt, logdir / "gt.gif", imgformat="rgba", verbose=False
        )

    # Estimate the length of the pendulum by backprop.

    parameters = []
    if args.optimize_length:
        length_est = torch.nn.Parameter(
            torch.tensor([0.5], device=device), requires_grad=False
        )
        lengthmodel = SimpleModel(length_est).to(device)
        parameters += list(lengthmodel.parameters())

    if args.optimize_gravity:
        gravity_est = torch.nn.Parameter(
            torch.tensor([5.0], device=device), requires_grad=False
        )
        gravitymodel = SimpleModel(gravity_est).to(device)
        parameters += list(gravitymodel.parameters())

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    optimizer = torch.optim.Adam(parameters, lr=1e-1)
    lossfn = torch.nn.MSELoss()

    losses = []
    best_loss_so_far = 1e6

    for e in trange(args.epochs):
        length_cur, gravity_cur = None, None
        length_cur = lengthmodel() if args.optimize_length else args.length
        gravity_cur = gravitymodel() if args.optimize_gravity else args.gravity
        pendulum_est = SimplePendulum(
            args.mass, args.radius, gravity_cur, length_cur, args.damping
        ).to(device)
        theta_init = torch.tensor([0.0, 3.0], device=device)
        ret = odeint(pendulum_est, theta_init, times)
        theta1 = ret[:, 0]
        theta2 = ret[:, 1]
        x = length_cur * theta1.sin()
        y = -length_cur * theta1.cos()
        z = torch.zeros_like(x)
        pos = torch.stack((x, y, z), dim=-1)
        imgs_est = []
        for i in range(numsteps):
            _vertices = vertices_gt.clone() + pos[i]
            rgba = renderer.forward(_vertices, faces_gt, textures)
            imgs_est.append(rgba)
        loss = sum(
            [
                lossfn(est, gt)
                for est, gt in zip(
                    imgs_est[:: args.compare_every], imgs_gt[:: args.compare_every]
                )
            ]
        ) / len(imgs_est[:: args.compare_every])
        # physics_loss = lossfn(x, x_gt) + lossfn(y, y_gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if loss.item() <= best_loss_so_far:
            best_loss_so_far = loss.item()
            if args.log:
                write_imglist_to_gif(
                    imgs_est, logdir / "best.gif", imgformat="rgba", verbose=False,
                )

        if args.log and e == 0:
            write_imglist_to_gif(
                imgs_est, logdir / "init.gif", imgformat="rgba", verbose=False,
            )
        if args.log and e == args.epochs - 1:
            write_imglist_to_gif(
                imgs_est, logdir / "opt.gif", imgformat="rgba", verbose=False,
            )

        tqdm.write(
            f"Loss: {loss.item():.5f}, "
            f"length_error: {(length_cur - args.length).abs().item():.5f}, "
            # f"gravity_error: {(gravity_cur - args.gravity).abs().item():.5f}"
        )

    if args.log:
        np.savetxt(logdir / "losses.txt", losses)

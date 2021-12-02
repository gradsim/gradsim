"""
Optimize parameters of a double-pendulum from video.
"""

import argparse
import math
from pathlib import Path

import torch
# Note: for this example, the Euler integration scheme works, while the
# adjoint-based integration scheme does not.
from torchdiffeq import odeint
from tqdm import tqdm, trange

from gradsim.bodies import DoublePendulum
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
        "--dtime", type=float, default=1 / 30, help="Simulation timestep size (sec)"
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=10.0,
        help="Acceleration due to gravity (m/s^2)",
    )
    parser.add_argument(
        "--length1", type=float, default=1.0, help="Length of the first bob (m)"
    )
    parser.add_argument(
        "--length2", type=float, default=1.0, help="Length of the second bob (m)"
    )
    parser.add_argument(
        "--mass1", type=float, default=1.0, help="Mass of the first bob (kg)"
    )
    parser.add_argument(
        "--mass2", type=float, default=1.0, help="Mass of the second bob (kg)"
    )
    parser.add_argument(
        "--radius", type=float, default=0.25, help="Radius of the bob (m)"
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=4,
        help="Apply loss every `--compare-every` frames.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs to run parameter optimization for",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=Path("sampledata/sphere.obj"),
        help="Path to template sphere mesh (.obj) file.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")

    args = parser.parse_args()

    if args.compare_every >= args.simsteps:
        raise ValueError(
            f"Arg --compare-every cannot be greater than or equal to {args.simsteps}."
        )

    # Device to run experiments on (MUST be CUDA-enabled, for render to work).
    device = "cuda:0"

    # Seed RNG for repeatability
    torch.manual_seed(args.seed)

    # Initialize the differentiable renderer.
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 8.0
    elevation = 0.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    # Load the template sphere mesh (for the bob)
    sphere = TriangleMesh.from_obj(args.template)
    vertices_gt = meshutils.normalize_vertices(
        sphere.vertices.unsqueeze(0), scale_factor=args.radius
    ).to(device)
    faces = sphere.faces.to(device).unsqueeze(0)
    textures_red = torch.cat(
        (
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )
    textures_blue = torch.cat(
        (
            torch.zeros(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )

    logdir = Path("cache/double_pendulum") / args.expid
    if args.log:
        logdir.mkdir(exist_ok=True)

    double_pendulum_gt = DoublePendulum(
        args.length1, args.length2, args.mass1, args.mass2, args.gravity
    )

    times = torch.arange(
        args.starttime,
        args.starttime + args.simsteps * args.dtime,
        args.dtime,
        device=device,
    )
    numsteps = times.numel()
    # Initial conditions
    y0 = torch.tensor([3 * math.pi / 7, 0, 3 * math.pi / 4, 0], device=device)

    # Perform numerical integration
    y_gt = odeint(double_pendulum_gt, y0, times)

    # Check that the calculation conserves total energy to within some tolerance
    edrift = 0.1
    # Total energy from initial conditions
    einit = double_pendulum_gt.compute_energy(y0)
    if (
        torch.max(torch.sum(torch.abs(double_pendulum_gt.compute_energy(y_gt) - einit)))
        > edrift
    ):
        print(f"[WARNING] Maximum energy drift of {edrift} exceeded!")
        # sys.exit(f"Maximum energy drift of {edrift} exceeded!")

    # Unpack z and theta as a function of time
    theta1_gt, theta2_gt = y_gt[:, 0], y_gt[:, 2]

    # Convert to Cartesian coordinates of the two bob positions
    x1 = double_pendulum_gt.length1 * theta1_gt.sin()
    y1 = -double_pendulum_gt.length1 * theta1_gt.cos()
    x2 = x1 + double_pendulum_gt.length2 * theta2_gt.sin()
    y2 = y1 - double_pendulum_gt.length2 * theta2_gt.cos()

    # GT positions of the first bob
    pos1_gt = torch.stack((x1, y1, torch.zeros_like(x1)), dim=-1)
    pos2_gt = torch.stack((x2, y2, torch.zeros_like(x2)), dim=-1)

    imgs1_gt = []
    imgs2_gt = []
    print("Rendering GT images...")
    for i in trange(numsteps):
        _vertices = vertices_gt.clone() + pos1_gt[i]
        rgba1 = renderer.forward(_vertices, faces, textures_red)
        imgs1_gt.append(rgba1)
        _vertices = vertices_gt.clone() + pos2_gt[i]
        rgba2 = renderer.forward(_vertices, faces, textures_blue)
        imgs2_gt.append(rgba2)

    if args.log:
        imgs_gt = [0.5 * (bob1 + bob2) for bob1, bob2 in zip(imgs1_gt, imgs2_gt)]
        write_imglist_to_gif(
            imgs_gt, logdir / "gt.gif", imgformat="rgba", verbose=False,
        )

    l1_est = torch.nn.Parameter(torch.tensor([1.0], device=device), requires_grad=False)
    l1model = SimpleModel(l1_est).to(device)
    l2_est = torch.nn.Parameter(torch.tensor([1.7], device=device), requires_grad=False)
    m1 = torch.tensor([args.mass1], device=device)
    m2 = torch.tensor([args.mass2], device=device)
    g = torch.tensor([args.gravity], device=device)
    l2model = SimpleModel(l2_est).to(device)

    optimizer = torch.optim.Adam(
        list(l1model.parameters()) + list(l2model.parameters()), lr=1e-2
    )
    lossfn = torch.nn.MSELoss()

    best_loss_so_far = 1e6

    for e in trange(args.epochs):
        l1_cur = l1model()
        l2_cur = l2model()
        double_pendulum = DoublePendulum(l1_cur, l2_cur, m1, m2, g)
        y = odeint(double_pendulum, y0, times)
        einit = double_pendulum.compute_energy(y0)
        if (
            torch.max(torch.sum(torch.abs(double_pendulum.compute_energy(y) - einit)))
            > edrift
        ):
            tqdm.write(f"[WARNING] Maximum energy drift of {edrift} exceeded!")
        theta1, theta2 = y[:, 0], y[:, 2]

        # Convert to Cartesian coordinates of the two bob positions
        x1 = l1_cur * theta1.sin()
        y1 = -l1_cur * theta1.cos()
        x2 = x1 + l2_cur * theta2.sin()
        y2 = y1 - l2_cur * theta2.cos()
        pos1 = torch.stack((x1, y1, torch.zeros_like(x1)), dim=-1)
        pos2 = torch.stack((x2, y2, torch.zeros_like(x2)), dim=-1)

        imgs1_est = []
        imgs2_est = []
        for i in range(numsteps):
            _vertices = vertices_gt.clone() + pos1[i]
            rgba1 = renderer.forward(_vertices, faces, textures_red)
            imgs1_est.append(rgba1)
            _vertices = vertices_gt.clone() + pos2[i]
            rgba2 = renderer.forward(_vertices, faces, textures_blue)
            imgs2_est.append(rgba2)

        loss = sum(
            [
                lossfn(est, gt)
                for est, gt in zip(
                    imgs1_est[:: args.compare_every], imgs1_gt[:: args.compare_every]
                )
            ]
            + [
                lossfn(est, gt)
                for est, gt in zip(
                    imgs2_est[:: args.compare_every], imgs2_gt[:: args.compare_every]
                )
            ]
        ) / (
            len(imgs1_est[:: args.compare_every])
            + len(imgs2_est[:: args.compare_every])
        )

        # loss = lossfn(pos1, pos1_gt) + lossfn(pos2, pos2_gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tqdm.write(
            f"Loss: {loss.item():.5f}, "
            f"l1_error: {(l1_cur - args.length1).abs().item():.5f}, "
            f"l2_error: {(l2_cur - args.length2).abs().item():.5f}"
        )

        if loss.item() <= best_loss_so_far:
            best_loss_so_far = loss.item()
            imgs_est = [0.5 * (bob1 + bob2) for bob1, bob2 in zip(imgs1_est, imgs2_est)]
            if args.log:
                write_imglist_to_gif(
                    imgs_est, logdir / "best.gif", imgformat="rgba", verbose=False,
                )

        if args.log and e == 0:
            imgs_est = [0.5 * (bob1 + bob2) for bob1, bob2 in zip(imgs1_est, imgs2_est)]
            write_imglist_to_gif(
                imgs_est, logdir / "init.gif", imgformat="rgba", verbose=False,
            )
        if args.log and e == args.epochs - 1:
            imgs_est = [0.5 * (bob1 + bob2) for bob1, bob2 in zip(imgs1_est, imgs2_est)]
            write_imglist_to_gif(
                imgs_est, logdir / "opt.gif", imgformat="rgba", verbose=False,
            )

# from matplotlib.patches import Circle
# # Plot a trail of the m2 bob's position for the last trail_secs seconds.
# trail_secs = 1
# # This corresponds to max_trail time points.
# max_trail = int(trail_secs / dt)

# def make_plot(i, logdir):
#     # Plot and save an image of the double pendulum config for time point i.

#     L1 = double_pendulum.length1
#     L2 = double_pendulum.length2
#     r = 0.05  # Plotted bob circle radius

#     # Pendulum rods
#     ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c="k")
#     # Circles representing the anchor point of rod 1, and bobs 1 and 2.
#     c0 = Circle((0, 0), r / 2, fc="k", zorder=10)
#     c1 = Circle((x1[i].detach().cpu().item(), y1[i].detach().cpu().item()),
#                 r, fc="b", ec="b", zorder=10)
#     c2 = Circle((x2[i].detach().cpu().item(), y2[i].detach().cpu().item()),
#                 r, fc="r", ec="r", zorder=10)
#     ax.add_patch(c0)
#     ax.add_patch(c1)
#     ax.add_patch(c2)

#     # The trail will be divided into ns segments and plotted as a fading line.
#     ns = 20
#     s = max_trail // ns

#     for j in range(ns):
#         imin = i - (ns - j) * s
#         if imin < 0:
#             continue
#         imax = imin + s + 1
#         # The fading looks better if we square the fractional length along a trail.
#         alpha = (j / ns) ** 2
#         ax.plot(
#             x2[imin:imax].detach().cpu().numpy(),
#             y2[imin:imax].detach().cpu().numpy(),
#             c="r",
#             solid_capstyle="butt",
#             lw=2,
#             alpha=alpha,
#         )

#     # Center the image on the fixed anchor point, and ensure the axes are equal.
#     ax.set_xlim(-L1 -L2 - r, L1 + L2 + r)
#     ax.set_ylim(-L1 -L2 - r, L1 + L2 + r)
#     ax.set_aspect("equal", adjustable="box")
#     plt.axis("off")
#     plt.savefig(logdir / f"{i:04d}.png", dpi=72)
#     plt.cla()


# # Make an image every di timesteps, corresponding to fps frames per second.
# fps = 10
# di = int(1 / fps / dt)
# fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
# ax = fig.add_subplot(111)

# logdir = Path("cache/double_pendulum", "debug-torch")
# logdir.mkdir(exist_ok=True)

# for i in trange(0, t.numel(), di):
#     # print(f"{i / di} / {t.size / di}")
#     make_plot(i, logdir)

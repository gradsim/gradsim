"""
Optimize parameters of a bouncing ball in 2D.
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm, trange

from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.utils import meshutils
from gradsim.utils.logging import write_imglist_to_gif


class BouncingBall2D:
    def __init__(
        self,
        pos=None,
        radius=1.0,
        theta=0.0,
        height=1.0,
        speed=1.0,
        gravity=-10.0,
        device="cpu",
    ):
        self.device = device
        self.radius = self.convert_to_tensor(radius)
        self.theta = self.convert_to_tensor(theta)
        self.height = self.convert_to_tensor(height)
        self.speed = self.convert_to_tensor(speed)

        if pos is None:
            self.position_initial = torch.zeros(
                2, dtype=self.radius.dtype, device=self.device
            )
        else:
            if not torch.is_tensor(pos):
                raise ValueError(
                    f"Input pos must be of type torch.Tensor. Got {type(pos)}"
                )
            self.position_initial = pos
        self.velocity_initial = torch.zeros(
            2, dtype=self.radius.dtype, device=self.device
        )
        self.position_initial[1] = self.height
        self.velocity_initial[0] = self.speed * self.theta.cos()
        self.velocity_initial[1] = self.speed * self.theta.sin()

        self.gravity = torch.zeros_like(self.position_initial)
        self.gravity[1] = gravity

        # Variables to store current state
        self.position_cur = self.position_initial.clone()
        self.velocity_cur = self.velocity_initial.clone()
        self.height_cur = self.height.clone()

        self.time = 0.0
        self.eps = 1e-16

    def convert_to_tensor(self, inp):
        if torch.is_tensor(inp):
            return inp.to(self.device)
        else:
            return torch.tensor([inp], dtype=torch.float32, device=self.device)

    def step(self, dtime):

        # Cache current velocity and position (needed to check collision)
        self.velocity_cur_cache = self.velocity_cur.clone()
        self.position_cur_cache = self.position_cur.clone()

        # Energy conservation: Leapfrog Method
        # https://adamdempsey90.github.io/python/bouncing_ball/bouncing_ball.html

        self.position_cur = self.position_cur + self.velocity_cur * dtime / 2
        self.velocity_cur = self.velocity_cur + self.gravity * dtime
        self.position_cur = self.position_cur + self.velocity_cur * dtime / 2

        # Handle collision with ground
        if self.position_cur[1] < self.radius:
            # Ball cannot sink below ground
            self.position_cur[1] = self.radius

            dtime_new = (self.radius - self.position_cur_cache[1]) / (
                self.velocity_cur_cache[1] + self.eps
            )
            self.velocity_cur[1] = -(
                self.velocity_cur_cache[1] + self.gravity[1] * dtime_new
            )
            self.position_cur[0] = (
                self.position_cur_cache[0] + self.velocity_cur[0] * dtime_new
            )
            self.time = self.time + dtime_new.item()

            # # Time of impact estimation
            # dtime_new = -self.position_cur_cache[1] / (self.velocity_cur_cache[1] + self.eps)
            # # Velocity at time of impact
            # self.velocity_cur[1] = -(self.velocity_cur_cache[1] + self.gravity[1] * dtime_new)
            # # Recompute the position, using this new velocity
            # self.position_cur[0] = self.position_cur_cache[0] + self.velocity_cur[0] * dtime_new

            # self.position_cur = self.position_cur + self.velocity_cur * (dtime_new - dtime) / 2
            # self.velocity_cur = self.velocity_cur + self.gravity * (dtime_new - dtime)
            # self.position_cur = self.position_cur + self.velocity_cur * (dtime_new - dtime) / 2
            # self.time = self.time + dtime_new  # Double check
        else:
            self.time = self.time + dtime

        # Update current height
        self.height_cur = self.position_cur[1]


class SimpleModel(torch.nn.Module):
    """A thin wrapper around a parameter, for convenient use with optim. """

    def __init__(self, param):
        super(SimpleModel, self).__init__()
        self.update = torch.nn.Parameter(torch.rand(param.shape))
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
        default=Path("cache/bounce2d"),
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
        default=100,
        help="Number of timesteps to run simulation for",
    )
    parser.add_argument(
        "--dtime", type=float, default=1 / 30, help="Simulation timestep size (sec)"
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

    # Detect and report autograd anomalies
    torch.autograd.set_detect_anomaly(True)

    # Seed RNG for repeatability
    torch.manual_seed(args.seed)

    device = "cuda:0"

    # # Hyperparams
    # dtime = 0.1  # 1 / 30
    # simsteps = 50
    # seed = 123
    # numepochs = 100
    # compare_every = 10
    # log = True
    # logdir = Path("cache/bounce2d")

    # Initialize the differentiable renderer.
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 15.0
    elevation = 0.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    # GT parameters
    position_initial_gt = torch.tensor([-7.5, 0.0], device=device)
    radius_gt = 0.75
    speed_gt = 1.0
    height_gt = 5.0
    gravity_gt = -9.0

    ball2d_gt = BouncingBall2D(
        pos=position_initial_gt,
        radius=radius_gt,
        height=height_gt,
        speed=speed_gt,
        gravity=gravity_gt,
        device=device,
    )

    sphere = TriangleMesh.from_obj(args.template)
    vertices_gt = meshutils.normalize_vertices(
        sphere.vertices.unsqueeze(0), scale_factor=radius_gt
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

    logdir = Path("cache/bounce2d") / args.expid
    if args.log:
        logdir.mkdir(exist_ok=True)

    traj_gt = []
    imgs_gt = []
    print("Rendering GT images...")
    for t in trange(args.simsteps):
        ball2d_gt.step(args.dtime)
        # traj.append((ball2d.position_cur[0].item(), ball2d.position_cur[1].item()))
        traj_gt.append(ball2d_gt.position_cur)
        pos = torch.zeros(3, dtype=vertices_gt.dtype, device=device)
        pos[0] = ball2d_gt.position_cur[0]
        pos[1] = ball2d_gt.position_cur[1]
        _vertices = vertices_gt.clone() + pos
        imgs_gt.append(renderer.forward(_vertices, faces, textures_red))

    if args.log:
        write_imglist_to_gif(
            imgs_gt, logdir / "gt.gif", imgformat="rgba", verbose=False,
        )

    # Parameters to estimate
    speed_est = torch.nn.Parameter(
        torch.tensor([3.0], device=device, requires_grad=True)
    )
    speedmodel = SimpleModel(speed_est).to(device)

    gravity_est = torch.nn.Parameter(
        torch.tensor([-10.0], device=device, requires_grad=True)
    )
    gravitymodel = SimpleModel(gravity_est).to(device)

    # Optimizer and loss function
    # optimizer = torch.optim.SGD(
    #     list(speedmodel.parameters()) + list(gravitymodel.parameters()), lr=1e-1
    # )
    optimizer = torch.optim.Adam(
        list(speedmodel.parameters()) + list(gravitymodel.parameters()), lr=1e-1
    )
    lossfn = torch.nn.MSELoss()

    for e in trange(args.epochs):
        speed_cur = speedmodel()
        gravity_cur = gravitymodel()
        ball2d = BouncingBall2D(
            pos=position_initial_gt,
            radius=radius_gt,
            height=height_gt,
            speed=speed_cur,
            gravity=gravity_cur,
            device=device,
        )
        vertices_gt = meshutils.normalize_vertices(
            sphere.vertices.unsqueeze(0), scale_factor=radius_gt
        ).to(device)
        traj_est = []
        imgs_est = []
        for t in range(args.simsteps):
            ball2d.step(args.dtime)
            pos = torch.zeros(3, dtype=vertices_gt.dtype, device=device)
            pos[0] = ball2d.position_cur[0]
            pos[1] = ball2d.position_cur[1]
            traj_est.append(ball2d.position_cur)
            _vertices = vertices_gt.clone() + pos
            imgs_est.append(renderer.forward(_vertices, faces, textures_red))
        # loss = sum(
        #     [lossfn(est, gt) for est, gt in zip(traj_est[:: compare_every], traj_gt[:: compare_every])]
        # ) / len(traj_est[:: compare_every])
        loss = sum(
            [
                lossfn(est, gt)
                for est, gt in zip(
                    imgs_est[:: args.compare_every], imgs_gt[:: args.compare_every]
                )
            ]
        ) / len(imgs_est[:: args.compare_every])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tqdm.write(
            f"Loss: {loss.item():.5f} "
            f"Speed error: {(speed_cur - speed_gt).item():.5f} "
            f"Gravity error: {(gravity_cur - gravity_gt).item():.5f}"
        )

        if args.log and e == 0:
            write_imglist_to_gif(
                imgs_est, logdir / "init.gif", imgformat="rgba", verbose=False,
            )
        if args.log and e == args.epochs - 1:
            write_imglist_to_gif(
                imgs_est, logdir / "opt.gif", imgformat="rgba", verbose=False,
            )

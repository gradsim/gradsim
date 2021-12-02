"""
Plots the loss landscape for a mass estimation scenario.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from gradsim.bodies import RigidBody
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.simulator import Simulator
from gradsim.utils import meshutils


class Model(torch.nn.Module):
    """Wrap masses into a torch.nn.Module, for ease of optimization. """

    def __init__(self, masses, uniform_density=False):
        super(Model, self).__init__()
        self.update = None
        if uniform_density:
            print("Using uniform density assumption...")
            self.update = torch.nn.Parameter(torch.rand(1) * 0.1)
        else:
            print("Assuming nonuniform density...")
            self.update = torch.nn.Parameter(torch.rand(masses.shape) * 0.1)
        self.masses = masses

    def forward(self):
        return torch.nn.functional.relu(self.masses + self.update)


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
        default=os.path.join("cache", "mass_loss_landscape"),
        help="Directory to store logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (for repeatability)"
    )
    parser.add_argument(
        "--infile",
        type=str,
        default=Path("sampledata/cube.obj"),
        help="Path to input mesh (.obj) file.",
    )
    parser.add_argument(
        "--simsteps",
        type=int,
        default=20,
        help="Number of steps to run simulation for.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to run optimization for.",
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=1,
        help="Apply loss every `--compare-every` frames.",
    )
    parser.add_argument(
        "--uniform-density",
        action="store_true",
        help="Whether to treat the object as having uniform density.",
    )
    parser.add_argument(
        "--force-magnitude",
        type=float,
        default=10.0,
        help="Magnitude of external force.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")

    args = parser.parse_args()

    if args.compare_every >= args.simsteps:
        raise ValueError(
            f"Arg --compare-every cannot be greater than or equal to {args.simsteps}."
        )

    # Seed RNG for repeatability.
    torch.manual_seed(args.seed)

    # We don't need gradients in this experiment
    torch.autograd.set_grad_enabled(False)

    # Device to store tensors on (MUST be CUDA-capable, for renderer to work).
    device = "cuda:0"

    # Load a body (from a triangle mesh obj file).
    mesh = TriangleMesh.from_obj(args.infile)
    vertices = meshutils.normalize_vertices(mesh.vertices.unsqueeze(0)).to(device)
    faces = mesh.faces.to(device).unsqueeze(0)
    textures = torch.cat(
        (
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )
    # masses_gt = torch.nn.Parameter(
    #     1 * torch.ones(vertices.shape[1], dtype=vertices.dtype, device=device),
    #     requires_grad=False,
    # )
    masses_gt = torch.nn.Parameter(
        torch.ones(vertices.shape[1], dtype=vertices.dtype, device=device),
        requires_grad=False,
    )
    body_gt = RigidBody(vertices[0], masses=masses_gt)

    # Create a force that applies gravity (g = 10 metres / second^2).
    gravity = ConstantForce(
        direction=torch.tensor([0.0, -1.0, 0.0]),
        magnitude=args.force_magnitude,
        device=device,
    )

    # Add this force to the body.
    body_gt.add_external_force(gravity, application_points=[0, 1])

    # Initialize the simulator with the body at the origin.
    sim_gt = Simulator([body_gt])

    # Initialize the renderer.
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 8.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    img_gt = []

    # Run the simulation.
    # writer = imageio.get_writer(outfile, mode="I")
    for i in trange(args.simsteps):
        sim_gt.step()
        # print("Body is at:", body.position)
        rgba = renderer.forward(
            body_gt.get_world_vertices().unsqueeze(0), faces, textures
        )
        img_gt.append(rgba)
    #     writer.append_data((255 * img).astype(np.uint8))
    # writer.close()

    lossfn = torch.nn.MSELoss()
    mass_estimates = []
    losses = []
    mass_errors = []

    est_masses = None
    initial_imgs = []
    initial_masses = None

    mass_interp = torch.linspace(0.1, 5, 500, dtype=vertices.dtype, device=device)

    for i in trange(mass_interp.numel()):
        masses_cur = torch.nn.Parameter(
            mass_interp[i]
            * torch.ones(vertices.shape[1], dtype=vertices.dtype, device=device),
            requires_grad=False,
        )
        body = RigidBody(vertices[0], masses=masses_cur)
        body.add_external_force(gravity, application_points=[0, 1])
        sim_est = Simulator([body])
        img_est = []
        for t in range(args.simsteps):
            sim_est.step()
            rgba = renderer.forward(
                body.get_world_vertices().unsqueeze(0), faces, textures
            )
            img_est.append(rgba)
            if i == 0:
                initial_imgs.append(rgba)  # To log initial guess.
        loss = sum(
            [
                lossfn(est, gt)
                for est, gt in zip(
                    img_est[:: args.compare_every], img_gt[:: args.compare_every]
                )
            ]
        ) / (len(img_est[:: args.compare_every]))
        # tqdm.write(
        #     f"Loss: {loss.item():.5f}, "
        #     f"Mass (err): {(masses_cur - masses_gt).abs().mean():.5f}"
        # )

        mass_estimates.append(mass_interp[i].item())
        mass_errors.append((masses_cur - masses_gt).abs().mean().item())
        losses.append(loss.item())

    # Save viz, if specified.
    if args.log:
        logdir = os.path.join(args.logdir, args.expid)
        os.makedirs(logdir, exist_ok=True)

        # Write metrics.
        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "mass_estimates.txt"), mass_estimates)
        np.savetxt(os.path.join(logdir, "mass_errors.txt"), mass_errors)

import argparse
import math
import os

# import imageio
import numpy as np
import torch
from tqdm import trange

from gradsim.bodies import RigidBody
from gradsim.engines import SemiImplicitEulerWithContacts
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.simulator import Simulator
from gradsim.utils import meshutils

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
        default=os.path.join("cache", "exp2_loss_landscape"),
        help="Directory to store logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (for repeatability)"
    )
    parser.add_argument(
        "--infile",
        type=str,
        default=os.path.join("sampledata", "cube.obj"),
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
    masses_gt = torch.nn.Parameter(
        torch.ones(vertices.shape[1], dtype=vertices.dtype, device=device),
        requires_grad=True,
    )
    position = torch.tensor([0.0, 4.0, 0.0], dtype=torch.float32, device=device)
    orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    restitution_gt = 0.8
    body = RigidBody(
        vertices[0],
        position=position,
        orientation=orientation,
        masses=masses_gt,
        restitution=restitution_gt,
    )

    # Create a force that applies gravity (g = 10 metres / second^2).
    # gravity = Gravity(device=device)
    force_magnitude = 10.0  # / vertices.shape[-2]
    gravity = ConstantForce(
        magnitude=force_magnitude,
        direction=torch.tensor([0.0, -1.0, 0.0]),
        device=device,
    )

    # Add this force to the body.
    body.add_external_force(gravity)

    sim_duration = 1.5
    fps = 30
    sim_substeps = 32
    dtime = (1 / 30) / sim_substeps
    sim_steps = int(sim_duration / dtime)
    render_every = sim_substeps

    # Initialize the simulator with the body at the origin.
    sim_gt = Simulator(
        bodies=[body], engine=SemiImplicitEulerWithContacts(), dtime=dtime
    )

    # Initialize the renderer.
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 10.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    # Run the simulation.
    # outfile = "cache/a.gif"
    # writer = imageio.get_writer(outfile, mode="I")
    imgs_gt = []
    for i in trange(sim_steps):
        sim_gt.step()
        # print("Body is at:", body.position)
        if i % render_every == 0:
            rgba = renderer.forward(
                body.get_world_vertices().unsqueeze(0), faces, textures
            )
            imgs_gt.append(rgba)
            # img = rgba[0].permute(1, 2, 0).detach().cpu().numpy()
            # writer.append_data((255 * img).astype(np.uint8))
    # writer.close()

    lossfn = torch.nn.MSELoss()

    mass_estimates = []
    mass_errors = []
    restitution_estimates = []
    restitution_errors = []
    losses = []

    mass_interp = torch.linspace(0.1, 5, 50, dtype=vertices.dtype, device=device)
    e_interp = torch.linspace(0.4, 1.0, 10, dtype=vertices.dtype, device=device)

    for i in trange(mass_interp.numel()):
        for j in trange(e_interp.numel()):
            masses_cur = torch.nn.Parameter(
                mass_interp[i]
                * torch.ones(vertices.shape[1], dtype=vertices.dtype, device=device),
                requires_grad=False,
            )
            e_cur = e_interp[j].item()
            body = RigidBody(vertices[0], masses=masses_cur, restitution=e_cur)
            sim_est = Simulator(
                bodies=[body], engine=SemiImplicitEulerWithContacts(), dtime=dtime
            )

            imgs_est = []
            for t in range(sim_steps):
                sim_est.step()
                if i % render_every == 0:
                    rgba = renderer.forward(
                        body.get_world_vertices().unsqueeze(0), faces, textures
                    )
                    imgs_gt.append(rgba)
                imgs_est.append(rgba)
            loss = sum(
                [
                    lossfn(est, gt)
                    for est, gt in zip(
                        imgs_est[:: args.compare_every], imgs_gt[:: args.compare_every]
                    )
                ]
            ) / (len(imgs_est[:: args.compare_every]))

            mass_estimates.append(mass_interp[i].item())
            mass_errors.append((masses_cur - masses_gt).abs().mean().item())
            restitution_estimates.append(e_cur)
            restitution_errors.append(abs(restitution_gt - e_cur))
            losses.append(loss.item())

    # Save viz, if specified.
    if args.log:
        logdir = os.path.join(args.logdir, args.expid)
        os.makedirs(logdir, exist_ok=True)

        # Write metrics.
        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "mass_estimates.txt"), mass_estimates)
        np.savetxt(os.path.join(logdir, "mass_errors.txt"), mass_errors)
        np.savetxt(
            os.path.join(logdir, "restitution_estimates.txt"), restitution_estimates
        )
        np.savetxt(os.path.join(logdir, "restitution_errors.txt"), restitution_errors)

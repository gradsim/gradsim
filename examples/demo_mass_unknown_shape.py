"""
Recover the mass and shape of an unknown object.
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

from gradsim.bodies import RigidBody
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.simulator import Simulator
from gradsim.utils import meshutils


class VertexModel(torch.nn.Module):
    """Wrap vertices into a torch.nn.Module, for optimization ease. """

    def __init__(self, vertices):
        super(VertexModel, self).__init__()
        self.update = torch.nn.Parameter(torch.rand(vertices.shape) * 0.001)
        self.vertices = vertices

    def forward(self):
        return self.vertices + self.update


class MassModel(torch.nn.Module):
    """Wrap masses into a torch.nn.Module, for ease of optimization. """

    def __init__(self, masses, uniform_density=False):
        super(MassModel, self).__init__()
        self.update = None
        if uniform_density:
            self.update = torch.nn.Parameter(torch.rand(1) * 0.1)
        else:
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
        default="cache/mass_unknown_shape",
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
        "--template",
        type=str,
        default=Path("sampledata/sphere.obj"),
        help="Path to input mesh (.obj) file.",
    )
    parser.add_argument(
        "--simsteps",
        type=int,
        default=20,
        help="Number of steps to run simulation for.",
    )
    parser.add_argument(
        "--shapeepochs",
        type=int,
        default=100,
        help="Number of epochs to run shape optimization for.",
    )
    parser.add_argument(
        "--massepochs",
        type=int,
        default=100,
        help="Number of epochs to run mass optimization for.",
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=10,
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

    # Device to store tensors on (MUST be CUDA-capable, for renderer to work).
    device = "cuda:0"

    # Load a body (from a triangle mesh obj file).
    mesh_gt = TriangleMesh.from_obj(args.infile)
    vertices_gt = meshutils.normalize_vertices(mesh_gt.vertices.unsqueeze(0)).to(device)
    faces_gt = mesh_gt.faces.to(device).unsqueeze(0)
    textures_gt = torch.cat(
        (
            torch.ones(1, faces_gt.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.ones(1, faces_gt.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces_gt.shape[1], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )
    mass_per_vertex = 1.0 / vertices_gt.shape[1]
    masses_gt = torch.nn.Parameter(
        mass_per_vertex
        * torch.ones(vertices_gt.shape[1], dtype=vertices_gt.dtype, device=device),
        requires_grad=False,
    )
    body_gt = RigidBody(vertices_gt[0], masses=masses_gt)

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
            body_gt.get_world_vertices().unsqueeze(0), faces_gt, textures_gt
        )
        img_gt.append(rgba)
    #     writer.append_data((255 * img).astype(np.uint8))
    # writer.close()

    # Load the template mesh (usually a sphere).
    mesh = TriangleMesh.from_obj(args.template)
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

    mass_per_vertex = 0.5 / vertices.shape[1]
    masses_est = torch.nn.Parameter(
        mass_per_vertex
        * torch.ones(vertices.shape[1], dtype=vertices.dtype, device=device),
        requires_grad=False,
    )
    massmodel = MassModel(masses_est, uniform_density=args.uniform_density)
    massmodel.to(device)
    optimizer = torch.optim.Adam(massmodel.parameters(), lr=5e-1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e1, momentum=0.9)
    lossfn = torch.nn.MSELoss()
    img_est = None  # Create a placeholder here, for global scope (useful in logging)
    masslosses = []
    est_masses = None
    initial_imgs = []
    initial_masses = None
    massmodel.train()
    for i in trange(args.massepochs):
        masses_cur = massmodel()
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
        tqdm.write(f"Mass Loss: {loss.item():.5f}, Mass (est): {masses_cur.mean():.5f}")
        # tqdm.write(f"Mass (GT): {masses_gt.mean():.5f}")
        masslosses.append(loss.item())
        est_masses = masses_est.clone().detach().cpu().numpy()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i == 40 or i == 80:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5

    massmodel.eval()
    shapemodel = VertexModel(vertices)
    shapemodel.to(device)
    shapelosses = []
    # optimizer = torch.optim.SGD(shapemodel.parameters(), lr=1e1, momentum=0.9)
    optimizer = torch.optim.Adam(shapemodel.parameters(), lr=1e-2)
    for i in trange(args.shapeepochs):
        masses_cur = massmodel()
        vertices_cur = shapemodel()
        body = RigidBody(vertices_cur[0], masses=masses_cur)
        body.add_external_force(gravity, application_points=[0, 1])
        sim_est = Simulator([body])
        img_est = []
        for t in range(args.simsteps):
            sim_est.step()
            rgba = renderer.forward(
                body.get_world_vertices().unsqueeze(0), faces, textures
            )
            img_est.append(rgba)
        loss = sum(
            [
                lossfn(est, gt)
                for est, gt in zip(
                    img_est[:: args.compare_every], img_gt[:: args.compare_every]
                )
            ]
        ) / len(img_est[:: args.compare_every])
        tqdm.write(
            f"Shape Loss: {loss.item():.5f}, Mass (est): {masses_cur.mean():.5f}"
        )
        shapelosses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save viz, if specified.
    if args.log:
        logdir = Path(args.logdir) / args.expid
        logdir.mkdir(exist_ok=True)

        # GT sim, Est sim
        initwriter = imageio.get_writer(logdir / "init.gif", mode="I")
        gtwriter = imageio.get_writer(logdir / "gt.gif", mode="I")
        estwriter = imageio.get_writer(logdir / "est.gif", mode="I")
        for gtimg, estimg, initimg in zip(img_gt, img_est, initial_imgs):
            gtimg = gtimg[0].permute(1, 2, 0).detach().cpu().numpy()
            estimg = estimg[0].permute(1, 2, 0).detach().cpu().numpy()
            initimg = initimg[0].permute(1, 2, 0).detach().cpu().numpy()
            gtwriter.append_data((255 * gtimg).astype(np.uint8))
            estwriter.append_data((255 * estimg).astype(np.uint8))
            initwriter.append_data((255 * initimg).astype(np.uint8))
        gtwriter.close()
        estwriter.close()
        initwriter.close()

        # Write metrics.
        np.savetxt(logdir / "masslosses.txt", masslosses)
        np.savetxt(logdir / "shapelosses.txt", shapelosses)
        np.savetxt(logdir / "masses.txt", est_masses)
        shape = shapemodel()[0].detach().cpu().numpy()
        np.savetxt(logdir / "shape.txt", shape)

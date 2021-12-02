"""
Estimate mass of a known rigid shape (primitives: cube, cylinder, sphere, etc.)

Use the first-cut dataset that Vikram generated.
"""

import argparse
import json
import os
import time

# import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from gradsim.assets.primitives import INT_TO_PRIMITIVE, get_primitive_obj
from gradsim.bodies import RigidBody
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.simulator import Simulator
from gradsim.utils import meshutils
from gradsim.utils.h5 import HDF5Dataset
# from gradsim.utils.logging import write_imglist_to_gif


class MassModel(torch.nn.Module):
    """Wrap masses into a torch.nn.Module, for ease of optimization. """

    def __init__(
        self, masses, uniform_density=False, minmass=0.1, maxmass=10.0, verbose=False
    ):
        super(MassModel, self).__init__()
        self.update = None
        self.masses = masses
        self.minmass = minmass
        self.maxmass = maxmass
        if uniform_density:
            if verbose:
                print("Using uniform density assumption...")
            # self.update = torch.nn.Parameter(torch.rand(1) * (minmass - maxmass) + maxmass)
            self.update = torch.nn.Parameter(torch.rand(1) * 0.1)
        else:
            if verbose:
                print("Assuming nonuniform density...")
            # self.update = torch.nn.Parameter(torch.rand(masses.shape) * (minmass - maxmass) + maxmass)
            self.update = torch.nn.Parameter(torch.rand(masses.shape) * 0.1)

    def forward(self):
        # return (self.update.repeat(self.masses.shape[0])).clamp(min=self.minmass, max=self.maxmass)
        return self.masses + self.update


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expid",
        type=str,
        default="default",
        help="Unique string identifier for the experiment.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.join("cache", "exp01"),
        help="Directory to store logs, for multiple runs of this experiment.",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default=os.path.join("cache", "dataset-rigid-exp1"),
        help="Directory containing the HDF5 dataset for the experiment.",
    )
    parser.add_argument(
        "--optiters",
        type=int,
        default=30,
        help="Number of iterations to optimize each object mass.",
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=5,
        help="Number of frames after which to compute pixel-wise MSE loss.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")

    args = parser.parse_args()

    device = "cuda:0"

    torch.manual_seed(42)

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    true_masses = []
    predicted_masses = []

    # Load 10 samples of only sequences
    dataset = HDF5Dataset(args.datadir, read_only_seqs=False)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=dataset.HDF5_collate_fn
    )

    for idx, out in enumerate(dataloader):

        starttime = time.time()

        print("Object:", idx)

        # First 800 images are used for training other baselines.
        if idx > 1:
            break

        out = next(iter(dataloader))

        (
            seqs,
            shape,
            init_pos,
            orientation,
            mass,
            fric,
            elas,
            color,
            scale,
            force_application_points,
            force_magnitude,
            force_direction,
            linear_velocity,
            angular_velocity,
        ) = out

        # # Run the simulation.
        # imgs_gt = []
        # # writer = imageio.get_writer("cache/a.gif", mode="I")
        # for i in trange(60):
        #     img = seqs[0, i]
        #     # writer.append_data(img.astype(np.uint8))
        #     imgs_gt.append(img.astype(np.uint8))
        # # writer.close()

        # Initialize the renderer
        image_size = 256
        camera_mode = "look_at"
        camera_distance = 8.0
        elevation = 30.0
        azimuth = 0.0
        # Initialize the renderer.
        renderer = SoftRenderer(
            image_size=image_size, camera_mode=camera_mode, device=device
        )
        renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

        sim_duration = 2.0  # seconds
        fps = 30  # frames per second
        sim_steps = int((sim_duration * fps) / 2)

        # obj = get_primitive_obj(shape[0])
        obj = get_primitive_obj(INT_TO_PRIMITIVE[shape[0]])
        # print(obj)
        mesh = TriangleMesh.from_obj(obj)
        vertices = (
            (
                meshutils.normalize_vertices(mesh.vertices)  # + \
                # torch.from_numpy(init_pos[i]).float().unsqueeze(0)
            )
            .to(device)
            .unsqueeze(0)
        )
        faces = mesh.faces.unsqueeze(0).to(device)
        textures = torch.cat(
            (
                color[0][0]
                / 255.0
                * torch.ones(
                    1, faces.shape[1], 2, 1, dtype=torch.float32, device=device
                ),
                color[0][1]
                / 255.0
                * torch.ones(
                    1, faces.shape[1], 2, 1, dtype=torch.float32, device=device
                ),
                color[0][2]
                / 255.0
                * torch.ones(
                    1, faces.shape[1], 2, 1, dtype=torch.float32, device=device
                ),
            ),
            dim=-1,
        )

        # print("GT mass:", mass[0], mass[0] * vertices.shape[-2])
        true_masses.append(mass[0])

        # (Uniform) Masses
        masses_gt = (float(mass[0])) * torch.nn.Parameter(
            torch.ones(vertices.shape[-2], dtype=vertices.dtype, device=device),
            requires_grad=True,
        )
        # Body
        body_gt = RigidBody(
            vertices[0],
            masses=masses_gt,
            # position=torch.from_numpy(init_pos[i]).float().to(device),
            orientation=torch.from_numpy(orientation[0]).float().to(device),
            friction_coefficient=float(fric[0]),
            restitution=float(elas[0]),
            # linear_velocity=torch.tensor(linear_velocity[i], device=device),
            # angular_velocity=torch.tensor(angular_velocity[i], device=device),
        )

        # inds = body_gt.vertices.argmin(1)
        # application_points = list(inds.view(-1).detach().cpu().numpy())
        # print("Est app points:", application_points)
        application_points = [force_application_points[0]]
        # print("True app points:", application_points)
        # Add a force
        force = ConstantForce(
            magnitude=force_magnitude[0],
            direction=torch.from_numpy(force_direction[0]).float().to(device),
            starttime=0.0,
            endtime=0.1,
            device=device,
        )
        body_gt.add_external_force(force, application_points=application_points)

        # Add gravity
        gravity = ConstantForce(
            magnitude=10.0, direction=torch.tensor([0, -1, 0]), device=device,
        )
        body_gt.add_external_force(gravity)

        sim_gt = Simulator([body_gt])

        # 2 seconds; 30 fps
        imgs_gt = []
        positions_gt = []
        with torch.no_grad():
            for t in range(sim_steps):
                sim_gt.step()
                rgba = renderer.forward(
                    body_gt.get_world_vertices().unsqueeze(0), faces, textures
                )
                imgs_gt.append(rgba)
                positions_gt.append(body_gt.position)

        masses_est = torch.nn.Parameter(
            (0.2) * torch.ones_like(masses_gt), requires_grad=True,
        )
        massmodel = MassModel(
            masses_est, uniform_density=True, minmass=1e-9, maxmass=1e9,
        )
        massmodel.to(device)

        # optimizer = torch.optim.Adam(massmodel.parameters(), lr=1)
        optimizer = torch.optim.SGD(massmodel.parameters(), lr=1e-3)
        lossfn = torch.nn.MSELoss()

        imgs_est = (
            None  # Create a placeholder here, for global scope (useful in logging)
        )
        positions_est = None
        losses = []
        est_masses = None
        initial_imgs = []
        initial_masses = None
        for i in range(args.optiters):
            masses_cur = massmodel()
            # print(masses_cur.mean())
            body = RigidBody(
                vertices=vertices[0],
                masses=masses_cur,
                orientation=torch.from_numpy(orientation[0]).float().to(device),
                friction_coefficient=float(fric[0]),
                restitution=float(elas[0]),
            )
            body.add_external_force(force, application_points=application_points)
            body.add_external_force(gravity)
            sim_est = Simulator([body])
            imgs_est = []
            positions_est = []
            for t in range(sim_steps):
                sim_est.step()
                rgba = renderer.forward(
                    body.get_world_vertices().unsqueeze(0), faces, textures
                )
                imgs_est.append(rgba)
                positions_est.append(body.position)
                if i == 0:
                    initial_imgs.append(rgba)  # To log initial guess.
            loss = sum(
                [
                    lossfn(est, gt)
                    for est, gt in zip(
                        positions_est[:: args.compare_every],
                        positions_gt[:: args.compare_every],
                    )
                ]
            ) / (len(positions_est[:: args.compare_every]))
            # if i % 5 == 0:
            tqdm.write(
                f"Loss: {loss.item():.5f}, "
                f"Mass (err): {(masses_cur - masses_gt).abs().mean():.5f}, "
                f"pred: {masses_cur.mean().item():.5f}"
            )
            #     # write_imglist_to_gif(
            #     #     imgs_est, "cache/exp01/est.gif", imgformat="rgba",
            #     # )
            # # tqdm.write(f"Mass (GT): {masses_gt.mean():.5f}")
            losses.append(loss.item())
            est_masses = masses_cur.clone().detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i == 40 or i == 80:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.5

            # write_imglist_to_gif(
            #     imgs_est, "cache/exp01/est.gif", imgformat="rgba",
            # )

        predicted_masses.append(est_masses.mean().item())

        print(f"Optimization time: {time.time() - starttime}")

    print("True masses:", true_masses)
    print("Predicted masses:", predicted_masses)

    if args.log:
        np.savetxt(os.path.join(logdir, "true_masses.txt"), true_masses)
        np.savetxt(os.path.join(logdir, "predicted_masses.txt"), predicted_masses)

    # # Sanity check, to ensure image rendering pipeline is sane

    # write_imglist_to_gif(
    #     imgs, "cache/regenerated.gif", imgformat="rgba", verbose=False
    # )

    # errors = []
    # errorsum = 0.
    # errcount = 0
    # for gt, pred in zip(imgs_gt, imgs):
    #     gt = torch.from_numpy(gt).float().to(device) / 255
    #     pred = ((pred[0].permute(1, 2, 0)) * 255)
    #     gt = gt.long()
    #     pred = pred.long()
    #     err = torch.nn.functional.mse_loss(pred[..., :3].float(), gt[..., :3].float())
    #     errors.append(err.item())
    #     errorsum += err.item()
    #     errcount += 1
    # print(errorsum / errcount)

    # # Run the simulation.
    # imgs_gt = []
    # # writer = imageio.get_writer("cache/a.gif", mode="I")
    # for i in trange(60):
    #     img = seqs[0, i]
    #     # writer.append_data(img.astype(np.uint8))
    #     imgs_gt.append(img.astype(np.uint8))
    # # writer.close()

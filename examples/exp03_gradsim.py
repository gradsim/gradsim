"""
Recover the mass and shape of an unknown object.
"""

import argparse
import json
import math
import os

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

from gradsim.bodies import RigidBody
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.renderutils.dibr.renderer import Renderer as DIBRenderer
from gradsim.renderutils.dibr.utils.sphericalcoord import get_spherical_coords_x
from gradsim.simulator import Simulator
from gradsim.utils import meshutils
from gradsim.utils.logging import write_imglist_to_gif


def write_meshes_to_dir(vertices_list, faces, dirname):
    os.makedirs(dirname, exist_ok=True)
    for i, vertices in enumerate(vertices_list):
        np.savetxt(os.path.join(dirname, f"vertices_{i:02d}.txt"), vertices)
    np.savetxt(os.path.join(dirname, "faces.txt"), faces)


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
        return self.masses + self.update


def compute_laplacian(vertices, faces):
    v1 = faces[:, 0].view(-1, 1)
    v2 = faces[:, 1].view(-1, 1)
    v3 = faces[:, 2].view(-1, 1)

    numvertices = vertices.shape[0]
    identity_indices = torch.arange(numvertices).view(-1, 1).to(v1.device)
    identity = torch.cat((identity_indices, identity_indices), dim=1).to(v1.device)
    identity = torch.cat((identity, identity))

    i_1 = torch.cat((v1, v2), dim=1)
    i_2 = torch.cat((v1, v3), dim=1)

    i_3 = torch.cat((v2, v1), dim=1)
    i_4 = torch.cat((v2, v3), dim=1)

    i_5 = torch.cat((v3, v2), dim=1)
    i_6 = torch.cat((v3, v1), dim=1)
    indices = torch.cat((identity, i_1, i_2, i_3, i_4, i_5, i_6), dim=0).t()
    values = torch.ones(indices.shape[1]).to(indices.device) * 0.5
    return torch.sparse.FloatTensor(
        indices, values, torch.Size([numvertices, numvertices])
    ).to(vertices)


def compute_laplacian_loss(vertices1, faces1, vertices2, faces2):
    laplacian1 = compute_laplacian(vertices1, faces1).to_dense()
    laplacian2 = compute_laplacian(vertices2, faces2).to_dense()
    return ((laplacian1 - laplacian2) ** 2).sum(-2).mean()


class LaplacianLoss(torch.nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x
        
class FlattenLoss(torch.nn.Module):
    def __init__(self, faces, average=False):
        super(FlattenLoss, self).__init__()
        self.nf = faces.size(0)
        self.average = average
        
        faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            for face in faces:
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(cos.ndimension())[1:])
        loss = (cos + 1).pow(2).sum(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss


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
        default=os.path.join("cache", "exp03-rigidshape"),
        help="Directory to store logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (for repeatability)"
    )
    parser.add_argument(
        "--infile",
        type=str,
        default=os.path.join("cache", "dataset-rigid-shape", "obj", "apple.obj"),
        help="Path to input mesh (.obj) file.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=os.path.join("sampledata", "sphere.obj"),
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
        default=1,
        help="Apply loss every `--compare-every` frames.",
    )
    parser.add_argument(
        "--non-uniform-density",
        action="store_true",
        help="Whether to treat the object as having non-uniform density.",
    )
    parser.add_argument(
        "--force-magnitude",
        type=float,
        default=1000.0,
        help="Magnitude of external force.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")
    parser.add_argument(
        "--log-every", type=int, default=10, help="How frequently to log gifs."
    )

    args = parser.parse_args()

    if args.compare_every >= args.simsteps:
        raise ValueError(
            f"Arg --compare-every cannot be greater than or equal to {args.simsteps}."
        )

    # Seed RNG for repeatability.
    torch.manual_seed(args.seed)

    torch.autograd.set_detect_anomaly(True)

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    # Device to store tensors on (MUST be CUDA-capable, for renderer to work).
    device = "cuda:0"

    print("Setting up the expt...")

    # Load a body (from a triangle mesh obj file).
    mesh_gt = TriangleMesh.from_obj(args.infile)
    vertices_gt = meshutils.normalize_vertices(mesh_gt.vertices.unsqueeze(0)).to(device)
    faces_gt = mesh_gt.faces.to(device).unsqueeze(0)
    textures_gt = torch.stack(
        (
            torch.ones(1, vertices_gt.shape[-2], dtype=torch.float32, device=device),
            torch.ones(1, vertices_gt.shape[-2], dtype=torch.float32, device=device),
            torch.zeros(1, vertices_gt.shape[-2], dtype=torch.float32, device=device),
        ),
        dim=-1,
    )
    # textures_gt = torch.cat(
    #     (
    #         torch.ones(1, faces_gt.shape[1], 2, 1, dtype=torch.float32, device=device),
    #         torch.ones(1, faces_gt.shape[1], 2, 1, dtype=torch.float32, device=device),
    #         torch.zeros(1, faces_gt.shape[1], 2, 1, dtype=torch.float32, device=device),
    #     ),
    #     dim=-1,
    # )
    # # uv_gt, textureimg_gt, lightparam_gt = None, None, None
    # uv_gt = get_spherical_coords_x(vertices_gt[0].cpu().numpy())
    # uv_gt = torch.from_numpy(uv_gt).to(device).float().unsqueeze(0) / 255.0
    # # Paint the mesh yellow
    # textureimg_gt = torch.zeros(1, 3, 128, 128, dtype=torch.float32, device=device)
    # textureimg_gt[:, 0, :, :] = 1.0
    # textureimg_gt[:, 1, :, :] = 1.0
    # # lightparam = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], device=device)
    # lightparam = 0.5 * torch.ones(9, device=device)

    # theta_gt = math.pi / 4  # Rotation to apply about the Y-axis
    # orientation_gt = torch.tensor([
    #     math.cos(theta_gt / 2), 0.0, math.sin(theta_gt / 2), 0.0
    # ], dtype=torch.float32, device=device)
    position_gt = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32, device=device)
    orientation_gt = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    # mass_per_vertex = 1.0 / vertices_gt.shape[1]
    mass_per_vertex = 1.0
    masses_gt = torch.nn.Parameter(
        mass_per_vertex
        * torch.ones(vertices_gt.shape[1], dtype=vertices_gt.dtype, device=device),
        requires_grad=False,
    )
    body_gt = RigidBody(
        vertices_gt[0],
        masses=masses_gt,
        position=position_gt,
        orientation=orientation_gt,
    )

    # Create a force that applies gravity (g = 10 metres / second^2).
    # normalized_magnitude_gt = args.force_magnitude / vertices_gt.shape[-2]
    gravity_gt = ConstantForce(
        direction=torch.tensor([0.0, -1.0, 0.0]),
        magnitude=args.force_magnitude,
        device=device,
    )

    # Add this force to the body.
    inds = vertices_gt.argmin(-2)[0]
    application_points = list(inds.view(-1).detach().cpu().numpy())
    body_gt.add_external_force(gravity_gt, application_points=application_points)

    # Initialize the simulator with the body at the origin.
    sim_gt = Simulator([body_gt])

    # Initialize the renderer.
    renderer = DIBRenderer(128, 128, mode="VertexColor")
    camera_distance = 8.0
    elevation = 30.0
    azimuth = 30.0
    renderer.set_look_at_parameters([90 - azimuth], [elevation], [camera_distance])
    # renderer = SoftRenderer(camera_mode="look_at", device=device)
    # renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    imgs_gt = []

    print("Running GT simulation...")
    # Run the simulation.
    with torch.no_grad():
        # writer = imageio.get_writer(outfile, mode="I")
        for i in trange(args.simsteps):
            sim_gt.step()
            # print("Body is at:", body.position)
            img_gt, alpha_gt, _ = renderer.forward(
                points=[
                    body_gt.get_world_vertices().unsqueeze(0), faces_gt[0].long()
                ],
                colors_bxpx3=textures_gt,
                # uv_bxpx2=uv_gt,
                # texture_bx3xthxtw=textureimg_gt,
                # lightparam=lightparam,
            )
            rgba = torch.cat((img_gt, alpha_gt), dim=-1)
            # rgba = renderer.forward(
            #     body_gt.get_world_vertices().unsqueeze(0), faces_gt, textures_gt
            # )
            imgs_gt.append(rgba)
        #     writer.append_data((255 * img).astype(np.uint8))
        # writer.close()

        if args.log:
            write_imglist_to_gif(imgs_gt, os.path.join(logdir, "gt.gif"), imgformat="dibr")

    # Load the template mesh (usually a sphere).
    mesh = TriangleMesh.from_obj(args.template)
    vertices = meshutils.normalize_vertices(mesh.vertices.unsqueeze(0)).to(device)
    faces = mesh.faces.to(device).unsqueeze(0)
    # uv = get_spherical_coords_x(vertices[0].cpu().numpy())
    # uv = torch.from_numpy(uv).to(device).float().unsqueeze(0) / 255.0
    # Paint the mesh yellow
    textures = torch.stack(
        (
            torch.ones(1, vertices.shape[-2], dtype=torch.float32, device=device),
            torch.ones(1, vertices.shape[-2], dtype=torch.float32, device=device),
            torch.zeros(1, vertices.shape[-2], dtype=torch.float32, device=device),
        ),
        dim=-1,
    )
    # textures = torch.cat(
    #     (
    #         torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
    #         torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
    #         torch.zeros(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
    #     ),
    #     dim=-1,
    # )

    # mass_per_vertex = 0.5 / vertices.shape[1]
    mass_per_vertex = 0.8
    masses_est = torch.nn.Parameter(
        mass_per_vertex
        * torch.ones(vertices.shape[1], dtype=vertices.dtype, device=device),
        requires_grad=False,
    )

    # normalized_magnitude = args.force_magnitude / vertices.shape[-2]
    gravity = ConstantForce(
        direction=torch.tensor([0.0, -1.0, 0.0]),
        magnitude=args.force_magnitude,
        device=device,
    )

    massmodel = MassModel(masses_est, uniform_density=not args.non_uniform_density)
    massmodel.to(device)
    shapemodel = VertexModel(vertices)
    shapemodel.to(device)

    # # 5e-2
    # optimizer = torch.optim.Adam(
    #     list(massmodel.parameters()) + list(shapemodel.parameters()), lr=5e-3
    # )
    lr = 5e-3
    optimizer_mass = torch.optim.Adam(list(massmodel.parameters()), lr=lr)
    optimizer_shape = torch.optim.Adam(list(shapemodel.parameters()), lr=5e-2)
    # optimizer = torch.optim.SGD(massmodel.parameters(), lr=1e-2, momentum=0.9)
    lossfn = torch.nn.MSELoss()
    laplossfn = LaplacianLoss(vertices[0].detach().cpu(), faces[0].detach().cpu())
    laplossfn.to(device)
    flatlossfn = FlattenLoss(faces[0].detach().cpu())
    imgs_est = None  # Create a placeholder here, for global scope (useful in logging)
    masslosses = []
    est_masses = None
    initial_imgs = []
    initial_masses = None
    massmodel.train()

    vertices_prev = vertices.clone()

    """
    Idea 2: get initial estimate of mass, by running mass opt for few epochs
    Then, fix this and get shape.
    """

    # args.massepochs = 30
    # # Freeze shapemodel for now
    # for param in shapemodel.parameters():
    #     param.requires_grad = False

    # for i in trange(args.massepochs):
    #     masses_cur = massmodel()
    #     vertices_cur = shapemodel()
    #     body = RigidBody(
    #         vertices_cur[0],
    #         masses=masses_cur,
    #         position=position_gt,
    #         orientation=orientation_gt,
    #     )
    #     # Add external force(s) to the body.
    #     inds = vertices.argmin(-2)
    #     application_points = list(inds.view(-1).detach().cpu().numpy())
    #     body.add_external_force(gravity, application_points=application_points)

    #     sim_est = Simulator([body])
    #     imgs_est = []
    #     for t in range(args.simsteps):
    #         sim_est.step()
    #         rgba = renderer.forward(
    #             body.get_world_vertices().unsqueeze(0), faces, textures
    #         )
    #         imgs_est.append(rgba)
    #         if i == 0:
    #             initial_imgs.append(rgba)  # To log initial guess.

    #     mseloss = sum(
    #         [
    #             lossfn(est, gt)
    #             for est, gt in zip(
    #                 imgs_est[:: args.compare_every], imgs_gt[:: args.compare_every]
    #             )
    #         ]
    #     ) / (len(imgs_est[:: args.compare_every]))
    #     laploss = 0.03 * laplossfn(vertices_cur[0]).mean()
    #     flatloss = 0.0003 * flatlossfn(vertices_cur).mean()

    #     loss = mseloss + laploss + flatloss

    #     tqdm.write(
    #         f"Total Loss: {loss.item():.5f}, "
    #         f"MSE: {mseloss.item():.5f}, Lap: {laploss.item():.5f}, Flat: {flatloss.item():.5f} "
    #         f"Mass error: {(masses_gt.mean() - masses_cur.mean()).abs():.5f}, "
    #         f"Mass (est): {masses_cur.mean():.5f}"
    #     )

    #     vertices_prev = vertices_cur

    #     if args.log and i % args.log_every == 0:
    #         write_imglist_to_gif(imgs_est, os.path.join(logdir, f"mass_{i:05d}.gif"))


    #     # tqdm.write(f"Mass (GT): {masses_gt.mean():.5f}")
    #     masslosses.append(loss.item())
    #     est_masses = masses_est.clone().detach().cpu().numpy()
    #     loss.backward()
    #     optimizer_mass.step()
    #     optimizer_mass.zero_grad()

    #     # if i == 40 or i == 80:
    #     #     for param_group in optimizer.param_groups:
    #     #         param_group["lr"] = param_group["lr"] * 0.5


    args.shapeepochs = 130
    # Unfreeze shapemodel
    for param in shapemodel.parameters():
        param.requires_grad = True
    # Freeze massmodel
    for param in massmodel.parameters():
        param.requires_grad = False

    # frame_of_interest = 0
    for i in trange(args.shapeepochs):
        masses_cur = torch.ones_like(masses_est)  # massmodel()
        vertices_cur = shapemodel()
        body = RigidBody(
            vertices_cur[0],
            masses=masses_cur,
            position=position_gt,
            orientation=orientation_gt,
        )
        # Add external force(s) to the body.
        inds = vertices.argmin(-2)[0]
        application_points = list(inds.view(-1).detach().cpu().numpy())
        body.add_external_force(gravity, application_points=application_points)

        sim_est = Simulator([body])
        imgs_est = []
        vertices_est = []
        for t in range(args.simsteps):
            sim_est.step()
            img, alpha, _ = renderer.forward(
                points=[
                    body.get_world_vertices().unsqueeze(0), faces[0].long()
                ],
                colors_bxpx3=textures,
            )
            rgba = torch.cat((img, alpha), dim=-1)
            # rgba = renderer.forward(
            #     body.get_world_vertices().unsqueeze(0), faces, textures
            # )
            imgs_est.append(rgba)
            vertices_est.append(body.get_world_vertices().detach().cpu().numpy())
            if i == 0:
                initial_imgs.append(rgba)  # To log initial guess.

        mseloss = sum(
            [
                lossfn(est, gt)
                for est, gt in zip(
                    imgs_est[::5], imgs_gt[::5]
                )
            ]
        ) / (len(imgs_est[::5]))
        laploss = 0.001 * laplossfn(vertices_cur[0]).mean()
        flatloss = 0.00001 * flatlossfn(vertices_cur).mean()

        loss = mseloss  # + laploss + flatloss

        tqdm.write(
            f"Total Loss: {loss.item():.5f}, "
            f"MSE: {mseloss.item():.5f}, Lap: {laploss.item():.5f}, Flat: {flatloss.item():.5f} "
            f"Mass error: {(masses_gt.mean() - masses_cur.mean()).abs():.5f}, "
            f"Mass (est): {masses_cur.mean():.5f}"
        )

        vertices_prev = vertices_cur

        if args.log and i % args.log_every == 0:
            write_imglist_to_gif(
                imgs_est, os.path.join(logdir, f"shape_{i:05d}.gif"), imgformat="dibr"
            )
            write_meshes_to_dir(
                vertices_est,
                faces[0].detach().cpu().numpy(),
                os.path.join(logdir, f"vertices_{i:05d}"),
            )


        # tqdm.write(f"Mass (GT): {masses_gt.mean():.5f}")
        masslosses.append(loss.item())
        est_masses = masses_est.clone().detach().cpu().numpy()
        loss.backward()
        optimizer_shape.step()
        optimizer_shape.zero_grad()

        # if i == 110:
        #     for pg in optimizer_shape.param_groups:
        #         pg["lr"] = pg["lr"] / 10.0

        # if i == 40 or i == 80:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = param_group["lr"] * 0.5



    """
    Idea 1: have a `masscycle` and `shapecycle` (i.e., for 20 epochs we optimize
    mass, for the next 20 shape, and so on). didn't work out so neat.
    """

    # masscycle = True

    # for i in trange(args.epochs):
    #     masses_cur = massmodel()
    #     vertices_cur = shapemodel()
    #     body = RigidBody(
    #         vertices_cur[0],
    #         masses=masses_cur,
    #         position=position_gt,
    #         orientation=orientation_gt,
    #     )
    #     # Add external force(s) to the body.
    #     inds = vertices.argmin(-2)
    #     application_points = list(inds.view(-1).detach().cpu().numpy())
    #     body.add_external_force(gravity, application_points=application_points)

    #     sim_est = Simulator([body])
    #     imgs_est = []
    #     for t in range(args.simsteps):
    #         sim_est.step()
    #         # img_pred, alpha, _ = renderer.forward(
    #         #     points=[
    #         #         body.get_world_vertices().unsqueeze(0), faces[0].long()
    #         #     ],
    #         #     uv_bxpx2=uv,
    #         #     texture_bx3xthxtw=textureimg,
    #         #     lightparam=lightparam,
    #         # )
    #         # rgba = torch.cat((img_pred, alpha), dim=-1)
    #         rgba = renderer.forward(
    #             body.get_world_vertices().unsqueeze(0), faces, textures
    #         )
    #         imgs_est.append(rgba)
    #         if i == 0:
    #             initial_imgs.append(rgba)  # To log initial guess.
    #     # laplacian_loss = 1e10 * compute_laplacian_loss(
    #     #     vertices_cur[0], faces[0], vertices_prev[0].detach(), faces[0]
    #     # )
    #     mseloss = sum(
    #         [
    #             lossfn(est, gt)
    #             for est, gt in zip(
    #                 imgs_est[:: args.compare_every], imgs_gt[:: args.compare_every]
    #             )
    #         ]
    #     ) / (len(imgs_est[:: args.compare_every]))

    #     laploss = 0.03 * laplossfn(vertices_cur[0]).mean()

    #     flatloss = 0.0003 * flatlossfn(vertices_cur).mean()

    #     loss = mseloss + laploss + flatloss

    #     tqdm.write(
    #         f"Total Loss: {loss.item():.5f}, "
    #         f"MSE: {mseloss.item():.5f}, Lap: {laploss.item():.5f}, Flat: {flatloss.item():.5f} "
    #         f"Mass error: {(masses_gt.mean() - masses_cur.mean()).abs():.5f}, "
    #         f"Mass (est): {masses_cur.mean():.5f}"
    #     )

    #     vertices_prev = vertices_cur

    #     if args.log and i % args.log_every == 0:
    #         write_imglist_to_gif(imgs_est, os.path.join(logdir, f"{i:05d}.gif"))


    #     # tqdm.write(f"Mass (GT): {masses_gt.mean():.5f}")
    #     masslosses.append(loss.item())
    #     est_masses = masses_est.clone().detach().cpu().numpy()
    #     loss.backward()
    #     if masscycle:
    #         optimizer_mass.step()
    #         optimizer_mass.zero_grad()
    #     else:
    #         optimizer_shape.step()
    #         optimizer_shape.zero_grad()

    #     if i % 20 == 0:
    #         masscycle = not masscycle
    #     # if i == 40 or i == 80:
    #     #     for param_group in optimizer.param_groups:
    #     #         param_group["lr"] = param_group["lr"] * 0.5







    # massmodel.eval()
    # shapemodel = VertexModel(vertices)
    # shapemodel.to(device)
    # shapelosses = []
    # # optimizer = torch.optim.SGD(shapemodel.parameters(), lr=1e1, momentum=0.9)
    # optimizer = torch.optim.Adam(shapemodel.parameters(), lr=1e-2)
    # for i in trange(args.shapeepochs):
    #     masses_cur = massmodel()
    #     vertices_cur = shapemodel()
    #     body = RigidBody(vertices_cur[0], masses=masses_cur)
    #     body.add_external_force(gravity, application_points=[0, 1])
    #     sim_est = Simulator([body])
    #     img_est = []
    #     for t in range(args.simsteps):
    #         sim_est.step()
    #         rgba = renderer.forward(
    #             body.get_world_vertices().unsqueeze(0), faces, textures
    #         )
    #         img_est.append(rgba)
    #     loss = sum(
    #         [
    #             lossfn(est, gt)
    #             for est, gt in zip(
    #                 img_est[:: args.compare_every], img_gt[:: args.compare_every]
    #             )
    #         ]
    #     ) / len(img_est[:: args.compare_every])
    #     tqdm.write(
    #         f"Shape Loss: {loss.item():.5f}, Mass (est): {masses_cur.mean():.5f}"
    #     )
    #     shapelosses.append(loss.item())
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    # # Save viz, if specified.
    # if args.log:
    #     logdir = Path(args.logdir) / args.expid
    #     logdir.mkdir(exist_ok=True)

    #     # GT sim, Est sim
    #     initwriter = imageio.get_writer(logdir / "init.gif", mode="I")
    #     gtwriter = imageio.get_writer(logdir / "gt.gif", mode="I")
    #     estwriter = imageio.get_writer(logdir / "est.gif", mode="I")
    #     for gtimg, estimg, initimg in zip(img_gt, img_est, initial_imgs):
    #         gtimg = gtimg[0].permute(1, 2, 0).detach().cpu().numpy()
    #         estimg = estimg[0].permute(1, 2, 0).detach().cpu().numpy()
    #         initimg = initimg[0].permute(1, 2, 0).detach().cpu().numpy()
    #         gtwriter.append_data((255 * gtimg).astype(np.uint8))
    #         estwriter.append_data((255 * estimg).astype(np.uint8))
    #         initwriter.append_data((255 * initimg).astype(np.uint8))
    #     gtwriter.close()
    #     estwriter.close()
    #     initwriter.close()

    #     # Write metrics.
    #     np.savetxt(logdir / "masslosses.txt", masslosses)
    #     np.savetxt(logdir / "shapelosses.txt", shapelosses)
    #     np.savetxt(logdir / "masses.txt", est_masses)
    #     shape = shapemodel()[0].detach().cpu().numpy()
    #     np.savetxt(logdir / "shape.txt", shape)

import argparse
import json
import math
import os

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

from gradsim import dflex as df
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.utils import meshutils
from gradsim.utils.logging import write_imglist_to_gif
from gradsim.utils.quaternion import quaternion_to_rotmat
from pxr import Usd, UsdGeom


def write_meshes_to_file(vertices_across_time, faces, dirname):
    os.makedirs(dirname, exist_ok=True)
    for i, vertices in enumerate(vertices_across_time):
        np.savetxt(os.path.join(dirname, f"{i:03d}.txt"), vertices)
    np.savetxt(os.path.join(dirname, "faces.txt"), faces)


class Model(torch.nn.Module):
    """Wrap params into a torch.nn.Module, for ease of optimization. """

    def __init__(self, params):
        super(Model, self).__init__()
        self.update = torch.nn.Parameter(torch.rand_like(params))
        self.params = params

    def forward(self):
        return self.params + self.update


def get_world_vertices(vertices, quaternion, translation):
    """Returns vertices transformed to world-frame. """
    rotmat = quaternion_to_rotmat(quaternion, scalar_last=True)
    return torch.matmul(rotmat, vertices.transpose(-1, -2)).transpose(
        -1, -2
    ) + translation.view(1, 3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for RNG."
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
        default=os.path.join("cache", "rigid-elasticity"),
        help="Directory to store experiment logs in.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=os.path.join("sampledata", "lowpoly", "box.obj"),
        help="Path to input mesh file (.obj format).",
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
        default=16,
        help="Number of sub-steps to integrate, per 1 `step` of the simulation.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training iterations."
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
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

    if args.log:
        print("LOGGING enabled!")
    else:
        print("LOGGING DISABLED!!")

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

    builder_gt = df.sim.ModelBuilder()
    obj = TriangleMesh.from_obj(args.mesh)
    vertices = meshutils.normalize_vertices(obj.vertices).to(device)
    points = vertices.detach().cpu().numpy()
    indices = list(obj.faces.numpy().reshape((-1)))

    mesh = df.sim.Mesh(points, indices)

    pos_gt = (0.0, 4.0, 0.0)
    rot_gt = df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.3)
    # rot_gt = (0.0, 0.0, 0.0, 0.0)
    vel_gt = (0.0, 2.0, 0.0)
    omega_gt = (0.0, 0.0, 0.0)
    scale_gt = (1.0, 1.0, 1.0)
    density_gt = 5.0
    ke_gt = 4900.0
    kd_gt = 15.0
    kf_gt = 990.0
    mu_gt = 0.77

    rigid_gt = builder_gt.add_rigid_body(
        pos=pos_gt, rot=rot_gt, vel=vel_gt, omega=omega_gt,
    )
    # scalefactor = 0.2
    # ke=1.0e4, kd=1000.0, kf=1000.0, mu=0.75,  # A nominal set of params
    shape_gt = builder_gt.add_shape_mesh(
        rigid_gt,
        mesh=mesh,
        scale=scale_gt,
        density=density_gt,
        ke=ke_gt,
        kd=kd_gt,
        kf=kf_gt,
        mu=mu_gt,
    )
    model_gt = builder_gt.finalize("cpu")
    model_gt.ground = True

    integrator = df.sim.SemiImplicitIntegrator()

    # Setup SoftRasterizer
    device = "cuda:0"
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 10.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    render_every = 60 * 4

    vertices = torch.from_numpy(np.array(points).astype(np.float32))
    faces = torch.from_numpy(np.asarray(indices).reshape((-1, 3)))
    textures = torch.cat(
        (
            torch.ones(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )

    imgs_gt = []
    positions_gt = []
    logvertices_gt = []
    faces_gt = None

    with torch.no_grad():

        sim_time = 0.0
        state_gt = model_gt.state()

        # construct contacts once at startup
        model_gt.collide(state_gt)

        for i in trange(sim_steps):

            # forward dynamics
            state_gt = integrator.forward(
                model_gt, state_gt, sim_dt
            )
            sim_time += sim_dt

            # render
            if i % render_every == 0 or i == sim_steps - 1:
                # with torch.no_grad():
                device = "cuda:0"
                # print(i, state.rigid_x, state.rigid_r, faces.unsqueeze(0).shape, textures.shape)
                # print(state.rigid_x.shape)
                # v_in = torch.from_numpy(np.asarray(model_gt.shape_geo_src[0].vertices)).float()
                # print(torch.allclose(v_in, vertices))
                vertices_current = get_world_vertices(
                    vertices, state_gt.rigid_r.view(-1), state_gt.rigid_x
                )
                rgba = renderer.forward(
                    vertices_current.unsqueeze(0).to(device),
                    faces.unsqueeze(0).to(device),
                    textures.to(device),
                )
                imgs_gt.append(rgba)
                positions_gt.append(state_gt.rigid_x)
                logvertices_gt.append(vertices_current.detach().cpu().numpy())

        if args.log:
            write_imglist_to_gif(imgs_gt, os.path.join(logdir, "gt.gif"))
            write_meshes_to_file(
                logvertices_gt,
                faces.detach().cpu().numpy(),
                os.path.join(logdir, "vertices_gt")
            )


    # """
    # Optimize for physical parameters.
    # """

    # # ke, kd, kf, mu
    # shape_material_guesses = (9500, 950, 950, 0.5)

    # builder = df.sim.ModelBuilder()
    # rigid = builder.add_rigid_body(
    #     pos=pos_gt, rot=rot_gt, vel=vel_gt, omega=omega_gt,
    # )
    # # scalefactor = 0.2
    # # ke=1.0e4, kd=1000.0, kf=1000.0, mu=0.75,  # A nominal set of params
    # shape = builder.add_shape_mesh(
    #     rigid_gt,
    #     mesh=mesh,
    #     scale=scale_gt,
    #     density=density_gt,
    #     ke=shape_material_guesses[0],
    #     kd=shape_material_guesses[1],
    #     kf=shape_material_guesses[2],
    #     mu=shape_material_guesses[3],
    # )
    # model = builder.finalize("cpu")
    # model.ground = True

    # # model.shape_materials.requires_grad = True
    # # params = torch.nn.Parameter(
    # #     model.shape_materials, requires_grad=True,
    # # )
    # params = Model(model.shape_materials)
    # optimizer = torch.optim.Adam(params.parameters(), lr=args.lr)
    # lossfn = torch.nn.MSELoss()

    # for e in trange(args.epochs):

    #     state = model.state()
    #     sim_time = 0.0
    #     model.collide(state)
    #     model.shape_materials = params()

    #     # clamp material params to reasonable range
    #     mat_min = torch.tensor((5000.0, 500.0, 500.0, 0.2))
    #     mat_max = torch.tensor((20000.0, 5000.0, 5000.0, 5.0))
    #     model.shape_materials = torch.max(torch.min(mat_max, model.shape_materials), mat_min)

    #     imgs = []
    #     positions = []
    #     logvertices = []

    #     for i in range(sim_steps):
    #         # forward dynamics
    #         state = integrator.forward(model, state, sim_dt)
    #         sim_time += sim_dt
    #         # render
    #         if i % render_every == 0 or i == sim_steps - 1:
    #             # with torch.no_grad():
    #             device = "cuda:0"
    #             vertices_current = get_world_vertices(
    #                 vertices, state.rigid_r.view(-1), state.rigid_x
    #             )
    #             rgba = renderer.forward(
    #                 vertices_current.unsqueeze(0).to(device),
    #                 faces.unsqueeze(0).to(device),
    #                 textures.to(device),
    #             )
    #             imgs.append(rgba)
    #             positions.append(state.rigid_x)
    #             logvertices.append(vertices_current.detach().cpu().numpy())

    #     if args.method == "gradsim":
    #         loss = sum(
    #             [
    #                 lossfn(est, gt)
    #                 for est, gt in zip(
    #                     imgs[:: args.compare_every], imgs_gt[:: args.compare_every]
    #                 )
    #             ]
    #         ) / (len(imgs[:: args.compare_every]))
    #     elif args.method == "physics-only":
    #         loss = sum(
    #             [
    #                 lossfn(est, gt)
    #                 for est, gt in zip(
    #                     positions[:: args.compare_every],
    #                     positions_gt[:: args.compare_every]
    #                 )
    #             ]
    #         ) / (len(imgs[:: args.compare_every]))

    #     tqdm.write(f"Loss: {loss.item():.5f} {model.shape_materials}")

    #     if args.log:
    #         write_meshes_to_file(
    #             logvertices,
    #             faces.detach().cpu().numpy(),
    #             os.path.join(logdir, f"vertices_{e:05d}")
    #         )

    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

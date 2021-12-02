import argparse
import math
import os

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

from gradsim import dflex as df
from gradsim.renderutils import SoftRenderer
from gradsim.utils.logging import write_imglist_to_gif
from pxr import Usd, UsdGeom

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true", help="Only run inference.")

    args = parser.parse_args()

    torch.manual_seed(42)

    torch.autograd.set_detect_anomaly(True)

    sim_duration = 1  # seconds
    sim_substeps = 16
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 200
    train_rate = 0.01  # 1.0/(sim_dt*sim_dt)

    phase_count = 8
    phase_step = math.pi / phase_count * 2.0
    phase_freq = 2.5

    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
    )

    builder = df.sim.ModelBuilder()

    mesh = Usd.Stage.Open("cache/usdassets/jellyfish.usda")
    geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/Icosphere/Icosphere"))

    points = geom.GetPointsAttr().Get()
    indices = geom.GetFaceVertexIndicesAttr().Get()
    counts = geom.GetFaceVertexCountsAttr().Get()

    face_materials = [-1] * len(counts)
    face_subsets = UsdGeom.Subset.GetAllGeomSubsets(geom)

    for i, s in enumerate(face_subsets):
        face_subset_indices = s.GetIndicesAttr().Get()

        for f in face_subset_indices:
            face_materials[f] = i

    active_material = 0
    active_scale = []

    def add_edge(f0, f1):
        if (
            face_materials[f0] == active_material
            and face_materials[f1] == active_material
        ):
            active_scale.append(1.0)
        else:
            active_scale.append(0.0)

    builder.add_cloth_mesh(
        pos=(0.0, 2.5, 0.0),
        rot=r,
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=points,
        indices=indices,
        edge_callback=add_edge,
        density=100.0,
    )

    model = builder.finalize("cpu")
    model.tri_lambda = 5000.0
    model.tri_ka = 5000.0
    model.tri_kd = 100.0
    model.tri_lift = 1000.0
    model.tri_drag = 0.0

    model.edge_ke = 20.0
    model.edge_kd = 1.0  # 2.5

    model.contact_ke = 1.0e4
    model.contact_kd = 0.0
    model.contact_kf = 1000.0
    model.contact_mu = 0.5

    model.particle_radius = 0.01
    model.ground = False
    model.gravity = torch.tensor((0.0, 0.0, 0.0))

    # training params
    target_pos = torch.tensor((4.0, 2.0, 0.0))

    rest_angle = model.edge_rest_angle

    # one fully connected layer + tanh activation
    network = torch.nn.Sequential(
        torch.nn.Linear(phase_count, model.edge_count, bias=False), torch.nn.Tanh(),
    )
    if args.inference:
        network = torch.load("cache/jellyfish/debug/model.pt")
        network.eval()
    else:
        network.train()

    activation_strength = math.pi * 0.3
    activation_scale = torch.tensor(active_scale)
    activation_penalty = 0.0

    integrator = df.sim.SemiImplicitIntegrator()

    render_time = 0

    # Setup SoftRasterizer
    device = "cuda:0"
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 10.0
    elevation = 0.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    faces = model.tri_indices
    textures = torch.cat(
        (
            torch.zeros(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
            torch.ones(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )

    # target_image = imageio.imread("cache/fem/debug/target_img_end.png")
    # target_image = torch.from_numpy(target_image).float().to(device) / 255.0
    # target_image = target_image.permute(2, 0, 1).unsqueeze(0)

    optimizer = torch.optim.SGD(network.parameters(), lr=train_rate, momentum=0.5)
    # optimizer = torch.optim.Adam(network.parameters(), lr=train_rate)

    epochs = 100
    validate_every = 10

    if args.inference:
        torch.autograd.set_grad_enabled(False)
        epochs = 1
        sim_duration = 25
        sim_dt = (1.0 / 60.0) / sim_substeps
        sim_steps = int(sim_duration / sim_dt)

    for e in trange(epochs):

        sim_time = 0.0

        state = model.state()

        loss = torch.zeros(1, requires_grad=True)
        # loss = None

        print_every = 60 * 16
        render_every = 60

        imgs = []

        for i in range(0, sim_steps):

            # build sinusoidal input phases
            phases = torch.zeros(phase_count)
            for p in range(phase_count):
                phases[p] = math.sin(phase_freq * (sim_time + p * phase_step))

            # compute activations (rest angles)
            activation = (network(phases)) * activation_strength * activation_scale
            model.edge_rest_angle = rest_angle + activation

            state = integrator.forward(model, state, sim_dt)
            sim_time += sim_dt * sim_substeps

            com_loss = torch.mean(state.u * model.particle_mass[:, None], 0)
            # com_loss = torch.mean(state.q, 0)
            act_loss = torch.norm(activation) * activation_penalty

            loss = loss - com_loss[1] - act_loss

            if i % render_every == 0 or i == sim_steps - 1:
                with torch.no_grad():
                    device = "cuda:0"
                    rgba = renderer.forward(
                        state.q.unsqueeze(0).to(device),
                        faces.unsqueeze(0).to(device),
                        textures.to(device),
                    )
                    imgs.append(rgba)

        if not args.inference:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        tqdm.write(f"Loss: {loss.item():.5}")

        render_time += 1
        if args.inference:
            filename = os.path.join("cache", "jellyfish", "debug", "inference")
        else:
            filename = os.path.join("cache", "jellyfish", "debug", f"{render_time:02d}")
        write_imglist_to_gif(imgs, f"{filename}.gif", imgformat="rgba", verbose=False)

        if args.inference:
            filename = os.path.join("cache", "jellyfish", "debug", "inference")
        else:
            filename = os.path.join(
                "cache", "jellyfish", "debug", f"last_frame_{render_time:02d}"
            )
        imageio.imwrite(
            f"{filename}.png",
            (imgs[-1][0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
                np.uint8
            ),
        )
        torch.save(network, "cache/jellyfish/debug/model.pt")

import math
from pathlib import Path

import torch
from tqdm import trange

from gradsim import dflex as df


class SimpleModel(torch.nn.Module):
    """A thin wrapper around a parameter, for convenient use with optim. """

    def __init__(self, param, activation=None):
        super(SimpleModel, self).__init__()
        self.update = torch.nn.Parameter(torch.rand(param.shape))
        self.param = param
        self.activation = activation

    def forward(self):
        out = self.param + self.update
        if self.activation is not None:
            return self.activation(out) + 1e-8
        return out


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(42)

    device = "cuda:0"

    sim_duration = 1.5  # seconds
    sim_substeps = 32
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    from gradsim.renderutils import SoftRenderer

    # from gradsim.renderutils.dibr.renderer import Renderer as DIBRenderer
    # from gradsim.renderutils.dibr.utils.sphericalcoord import get_spherical_coords_x
    from gradsim.utils.logging import write_imglist_to_gif

    renderer = SoftRenderer(camera_mode="look_at", device=device)
    # renderer = DIBRenderer(256, 256, mode="VertexColor")
    camera_distance = 5.0
    elevation = 0.0
    azimuth = 0.0
    # renderer.set_look_at_parameters([90 - azimuth], [elevation], [camera_distance])
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)  # For SoftRas

    render_steps = 60 * 4

    train_iters = 100
    train_rate = 0.01

    height = 1.5

    particle_inv_mass_gt = None
    particle_velocity_gt = None

    with torch.no_grad():
        builder = df.sim.ModelBuilder()
        builder.add_cloth_grid(
            pos=(-2.0, height, 0.0),
            rot=df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 1.04),
            vel=(1.0, 0.0, 0.0),
            dim_x=20,
            dim_y=10,
            cell_x=0.1,
            cell_y=0.1,
            mass=1.0,
        )  # , fix_left=True, fix_right=True, fix_top=True, fix_bottom=True)

        attach0 = 0
        attach1 = 20

        anchor0 = builder.add_particle(
            pos=builder.particle_positions[attach0] - (1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=0.0,
        )
        anchor1 = builder.add_particle(
            pos=builder.particle_positions[attach1] + (1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=0.0,
        )

        builder.add_spring(anchor0, attach0, 10000.0, 1000.0, 0)
        builder.add_spring(anchor1, attach1, 10000.0, 1000.0, 0)

        model = builder.finalize("cpu")
        model.tri_lambda = 10000.0
        model.tri_ka = 10000.0
        model.tri_kd = 100.0

        model.contact_ke = 1.0e4
        model.contact_kd = 1000.0
        model.contact_kf = 1000.0
        model.contact_mu = 0.5

        model.particle_radius = 0.01
        model.ground = False

        # print(model.particle_inv_mass)
        # model.particle_inv_mass = (torch.arange(model.particle_inv_mass.numel()).float() + 1.0) / 20.0
        # model.particle_inv_mass[-1] = 0.0
        # model.particle_inv_mass[-2] = 0.0
        numparticles = model.particle_inv_mass.numel()
        model.particle_inv_mass[: numparticles // 2] = torch.rand_like(
            model.particle_inv_mass[: numparticles // 2]
        )
        model.particle_inv_mass[numparticles // 2 :] = 2 * torch.rand_like(
            model.particle_inv_mass[numparticles // 2 :]
        )
        model.particle_inv_mass[-1] = 0.0
        model.particle_inv_mass[-2] = 0.0
        # print(model.particle_inv_mass)

        particle_inv_mass_gt = model.particle_inv_mass.clone()
        particle_velocity_gt = model.particle_velocity.clone()

        integrator = df.sim.SemiImplicitIntegrator()

        sim_time = 0.0
        state = model.state()

        faces = model.tri_indices
        textures = torch.cat(
            (
                torch.ones(
                    1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                ),
                torch.ones(
                    1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                ),
                torch.zeros(
                    1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                ),
            ),
            dim=-1,
        )
        # texture_img = imageio.imread(Path("sampledata/texture.png"))
        # texture_img = torch.from_numpy(texture_img).to(device).float() / 255.0
        # texture_img = texture_img.permute(2, 0, 1).unsqueeze(0)
        # uv = get_spherical_coords_x(state.q.detach().cpu().numpy())
        # uv = torch.from_numpy(uv).cuda().unsqueeze(0)

        # run simulation
        imgs_gt = []
        for i in trange(0, sim_steps):
            state = integrator.forward(model, state, sim_dt)
            sim_time += sim_dt

            com_pos = torch.mean(state.q, 0)
            com_vel = torch.mean(state.u, 0)

            if i % render_steps == 0:
                # rgb, alpha, _ = renderer(
                #     points=[state.q.unsqueeze(0).to(device), faces.long()],
                #     colors_bxpx3=torch.ones(1, state.q.shape[-2], 3).float().cuda(),
                #     # uv_bxpx2=uv,
                #     # texture_bx3xthxtw=texture_img,
                # )
                # import matplotlib.pyplot as plt
                # plt.imshow(rgb[0].detach().cpu().numpy())
                # plt.show()
                # sys.exit(0)
                # imgs_gt.append(rgb)

                # SoftRas
                rgba = renderer.forward(
                    state.q.unsqueeze(0).to(device),
                    faces.unsqueeze(0).to(device),
                    textures.to(device),
                )
                imgs_gt.append(rgba)

        cloth_path = Path("cache/cloth")
        cloth_path.mkdir(exist_ok=True)

        write_imglist_to_gif(
            imgs_gt, cloth_path / "gt.gif", imgformat="rgba", verbose=False
        )

    init_inv_mass = particle_inv_mass_gt.clone()
    init_inv_mass[-1] = 0.0
    init_inv_mass[-2] = 0.0
    massmodel = SimpleModel(
        # particle_inv_mass_gt + 50 * torch.rand_like(particle_inv_mass_gt),
        init_inv_mass,
        activation=torch.nn.functional.relu,
    )
    velocitymodel = SimpleModel(
        -0.01 * torch.ones_like(particle_velocity_gt), activation=None
    )
    epochs = 50
    save_gif_every = 1
    compare_every = 1

    optimizer = torch.optim.Adam(list(velocitymodel.parameters()), lr=5e-2)
    lossfn = torch.nn.MSELoss()

    for e in range(epochs):

        if e in [10, 15, 20, 30]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] / 2

        builder = df.sim.ModelBuilder()
        builder.add_cloth_grid(
            pos=(-2.0, height, 0.0),
            rot=df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 1.04),
            vel=(1.0, 0.0, 0.0),
            dim_x=20,
            dim_y=10,
            cell_x=0.1,
            cell_y=0.1,
            mass=1.0,
        )  # , fix_left=True, fix_right=True, fix_top=True, fix_bottom=True)

        attach0 = 0
        attach1 = 20

        anchor0 = builder.add_particle(
            pos=builder.particle_positions[attach0] - (1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=0.0,
        )
        anchor1 = builder.add_particle(
            pos=builder.particle_positions[attach1] + (1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=0.0,
        )

        builder.add_spring(anchor0, attach0, 10000.0, 1000.0, 0)
        builder.add_spring(anchor1, attach1, 10000.0, 1000.0, 0)

        model = builder.finalize("cpu")
        model.tri_lambda = 10000.0
        model.tri_ka = 10000.0
        model.tri_kd = 100.0

        model.contact_ke = 1.0e4
        model.contact_kd = 1000.0
        model.contact_kf = 1000.0
        model.contact_mu = 0.5

        model.particle_radius = 0.01
        model.ground = False

        # model.particle_inv_mass = massmodel()
        # # print(model.particle_inv_mass.max(), model.particle_inv_mass.min(), model.particle_inv_mass.mean())
        # model.particle_inv_mass[-1] = 0.0
        # model.particle_inv_mass[-2] = 0.0

        model.particle_velocity = velocitymodel()

        integrator = df.sim.SemiImplicitIntegrator()

        sim_time = 0.0
        state = model.state()

        faces = model.tri_indices
        textures = torch.cat(
            (
                torch.ones(
                    1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                ),
                torch.ones(
                    1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                ),
                torch.zeros(
                    1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device
                ),
            ),
            dim=-1,
        )

        # run simulation
        imgs = []
        for i in trange(0, sim_steps):
            state = integrator.forward(model, state, sim_dt)
            sim_time += sim_dt

            com_pos = torch.mean(state.q, 0)
            com_vel = torch.mean(state.u, 0)

            if i % render_steps == 0:
                rgba = renderer.forward(
                    state.q.unsqueeze(0).to(device),
                    faces.unsqueeze(0).to(device),
                    textures.to(device),
                )
                imgs.append(rgba)
        loss = sum(
            [
                lossfn(est, gt)
                for est, gt in zip(imgs[::compare_every], imgs_gt[::compare_every])
            ]
        ) / len(imgs[::compare_every])
        print("Loss:", loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (e % save_gif_every == 0) or (e == epochs - 1):
            write_imglist_to_gif(
                imgs, f"cache/cloth/{e:05d}.gif", imgformat="rgba", verbose=False
            )
            # np.savetxt(f"cache/cloth/mass_{e:05d}.txt", model.particle_inv_mass.detach().cpu().numpy())

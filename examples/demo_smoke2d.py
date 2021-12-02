from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange


def rollrow():
    return lambda x, shift: torch.roll(x, shift, 0)


def rollcol():
    return lambda x, shift: torch.roll(x, shift, 1)


def project(vx, vy, params):
    """Project the velocity field to be approximately mass-conserving, using a few
    iterations of Gauss-Siedel. """
    p = torch.zeros(vx.shape, dtype=vx.dtype, device=vy.device)
    h = 1.0 / vx.shape[0]
    div = (
        -0.5
        * h
        * (rollrow()(vx, -1) - rollrow()(vx, 1) + rollcol()(vy, -1) - rollcol()(vy, 1))
    )

    for k in range(6):
        p = (
            div
            + rollrow()(p, 1)
            + rollrow()(p, -1)
            + rollcol()(p, 1)
            + rollcol()(p, -1)
        ) / 4.0

    vx = vx - 0.5 * (rollrow()(p, -1) - rollrow()(p, 1)) / h
    vy = vy - 0.5 * (rollcol()(p, -1) - rollcol()(p, 1)) / h
    return vx, vy


def advect(f, vx, vy, params):
    """Move field f according to velocities vx, vy using an implicit Euler
    integrator. """
    rows, cols = f.shape
    cell_ys, cell_xs = torch.meshgrid(torch.arange(rows), torch.arange(cols))
    cell_xs = torch.transpose(cell_xs, 0, 1).float().to(params["device"])
    cell_ys = torch.transpose(cell_ys, 0, 1).float().to(params["device"])
    center_xs = (cell_xs - vx).flatten()
    center_ys = (cell_ys - vy).flatten()

    # Compute indices of source cells
    left_ix = torch.floor(center_xs).long()
    top_ix = torch.floor(center_ys).long()
    rw = center_xs - left_ix.float()  # Relative weight of right-hand cells
    bw = center_ys - top_ix.float()  # Relative weight of bottom cells
    left_ix = torch.remainder(left_ix, rows)  # Wrap around edges of simulation
    right_ix = torch.remainder(left_ix + 1, rows)
    top_ix = torch.remainder(top_ix, cols)
    bot_ix = torch.remainder(top_ix + 1, cols)

    # A linearly-weighted sum of the 4 surrounding cells
    flat_f = (1 - rw) * (
        (1 - bw) * f[left_ix, top_ix] + bw * f[left_ix, bot_ix]
    ) + rw * ((1 - bw) * f[right_ix, top_ix] + bw * f[right_ix, bot_ix])

    return torch.reshape(flat_f, (rows, cols))


def forward(iteration, smoke, vx, vy, output, params):
    for t in range(1, params["steps"]):
        vx_updated = advect(vx, vx, vy, params)
        vy_updated = advect(vy, vx, vy, params)
        vx, vy = project(vx_updated, vy_updated, params)
        smoke = advect(smoke, vx, vy, params)
        if output:
            smoke2d_path = Path("cache/smoke2d")
            smoke2d_path.mkdir(parents=True, exist_ok=True)
            plt.imsave(
                smoke2d_path / f"{t:03d}.png", 255 * smoke.cpu().detach().numpy()
            )
    return smoke


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(231)

    numiters = 50
    gridsize = 110
    dx = 1.0 / gridsize
    lr = 1e-1
    printevery = 1

    params = {}
    params["device"] = device
    params["steps"] = 100

    # initial_smoke_img = (
    #     imageio.imread(Path("sampledata/init_smoke.png"))[:, :, 0] / 255.0
    # )
    # target_smoke_img = (
    #     imageio.imread(Path("sampledata/peace.png"))[::2, ::2, 3] / 255.0
    # )
    target_smoke_img = imageio.imread(Path("sampledata/smoke_target.png")) / 255.0

    vx = torch.nn.Parameter(
        torch.zeros(
            gridsize, gridsize, dtype=torch.float32, device=device, requires_grad=True
        )
    )
    vy = torch.nn.Parameter(
        torch.zeros(
            gridsize, gridsize, dtype=torch.float32, device=device, requires_grad=True
        )
    )
    # initial_smoke = torch.tensor(initial_smoke_img, device=device, dtype=torch.float32)
    target = torch.tensor(target_smoke_img, device=device, dtype=torch.float32)

    initial_smoke = torch.zeros_like(target)
    initial_smoke[2 * gridsize // 3 :, :] = 1.0

    optimizer = torch.optim.Adam([vx] + [vy], lr=lr)

    for iteration in trange(numiters):
        # t = time.time()
        smoke = forward(
            iteration, initial_smoke, vx, vy, iteration == (numiters - 1), params
        )
        loss = torch.nn.functional.mse_loss(smoke, target)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if iteration % printevery == 0:
            tqdm.write(f"Iter {iteration} Loss: {loss.item():.8}")

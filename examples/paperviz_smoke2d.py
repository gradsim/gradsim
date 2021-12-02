from pathlib import Path

import colorsys
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import vapeplot
from scipy.interpolate import interp1d
from tqdm import tqdm, trange


def get_colors(palette, num):
    # palette = 'vaporwave'
    # palette = 'cool'
    colors = np.array([colorsys.rgb_to_hsv(*tuple(int(c[i:i+2], 16) for i in (1, 3, 5)))[0] for c in vapeplot.palette(palette)])
    f = interp1d(np.arange(len(colors)), colors)
    return f(np.arange(num)/(num-1)*(len(colors)-1))


def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv


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
            smoke2d_path = Path(f"cache/suppl-viz/smoke2d/{iteration:03d}")
            smoke2d_path.mkdir(parents=True, exist_ok=True)

            smokeimg = (smoke - smoke.min()) / (smoke.max() - smoke.min())
            smokeimg = smokeimg.detach().cpu().numpy()
            # smokeimg = 255 * smokeimg
            # smokeimg = smokeimg.astype(np.uint8)

            pal = vapeplot.palette("cool")
            pal = ["#000000", "#FFFFFF"]
            cmap = matplotlib.colors.ListedColormap(pal)
            # print(pal)
            plt.imsave(
                smoke2d_path / f"{t:03d}.png", 255 * smokeimg,  # 255 * smoke.cpu().detach().numpy(),
                cmap=cmap,
            )
    return smoke


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(231)

    numiters = 100
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

    save_every = 10

    for iteration in trange(numiters):
        # t = time.time()
        smoke = forward(
            iteration, initial_smoke, vx, vy, (iteration % save_every) == 0, params
        )
        # smoke = forward(
        #     iteration, initial_smoke, vx, vy, iteration == (numiters - 1), params
        # )
        loss = torch.nn.functional.mse_loss(smoke, target)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if iteration % printevery == 0:
            tqdm.write(f"Iter {iteration} Loss: {loss.item():.8}")

"""
DIB-R: Sanity check
Uses DIB-R to optimize the vertices a given mesh to match a rendered image.
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

from gradsim.renderutils import TriangleMesh
from gradsim.renderutils.dibr.renderer import Renderer as DIBRenderer
from gradsim.renderutils.dibr.utils.sphericalcoord import get_spherical_coords_x

# Example script that uses SoftRas to deform a sphere mesh to aproximate
# the image of a banana.


class Model(torch.nn.Module):
    """Wrap vertices into an nn.Module, for optimization. """

    def __init__(self, vertices):
        super(Model, self).__init__()
        self.update = torch.nn.Parameter(torch.rand(vertices.shape) * 0.001)
        self.verts = vertices

    def forward(self):
        return self.update + self.verts


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
    laplacian1 = compute_laplacian(vertices1, faces1)
    laplacian2 = compute_laplacian(vertices2, faces2)
    return ((laplacian1 - laplacian2) ** 2).sum(-2).mean()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of iterations to run optimization for.",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization steps."
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=["vc", "lam", "sh"],
        default="vc",
        help='Type of color handling to used in renderer. "vc" uses VertexColor mode. '
        '"lam" uses Lambertian mode. "sh" uses SphericalHarmonics.',
    )
    args = parser.parse_args()

    # Device to store tensors on. Must be a CUDA device.
    device = "cuda:0"

    # Initialize the soft rasterizer.
    if args.renderer == "vc":
        renderer = DIBRenderer(256, 256, mode="VertexColor")
    elif args.renderer == "lam":
        renderer = DIBRenderer(256, 256, mode="Lambertian")
    elif args.renderer == "sh":
        renderer = DIBRenderer(256, 256, mode="SphericalHarmonics")
    else:
        raise ValueError(
            'Renderer mode must be one of ["VertexColor", "Lambertian"'
            f' or "SphericalHarmonics"]Got {args.renderer} instead.'
        )

    # Camera settings.
    camera_distance = (
        2.0  # Distance of the camera from the origin (i.e., center of the object)
    )
    elevation = 30.0  # Angle of elevation
    azimuth = 0.0  # Azimuth angle

    # Directory in which sample data is located.
    DATA_DIR = Path(__file__).parent / "sampledata"

    # Directory in which logs (gifs) are saved.
    logdir = Path(__file__).parent / "cache" / "dibr"
    logdir.mkdir(exist_ok=True)

    # Read in the input mesh. TODO: Add filepath as argument.
    mesh = TriangleMesh.from_obj(DATA_DIR / "dibr_sphere.obj")

    # Output filename to write out a rendered .gif to, showing the progress of optimization.
    progressfile = logdir / "vertex_optimization_progress.gif"
    # Output filename to write out a rendered .gif file to, rendering the optimized mesh.
    outfile = logdir / "vertex_optimization_output.gif"

    # Extract the vertices, faces, and texture the mesh (currently color with white).
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()
    # Initialize all faces to yellow (to color the banana)!
    textures = torch.stack(
        (
            torch.ones(1, vertices.shape[-2], dtype=torch.float32, device=device),
            torch.ones(1, vertices.shape[-2], dtype=torch.float32, device=device),
            torch.zeros(1, vertices.shape[-2], dtype=torch.float32, device=device),
        ),
        dim=-1,
    )

    uv, texture_img, lightparam = None, None, None
    if args.renderer in ["lam", "sh"]:
        uv = get_spherical_coords_x(vertices[0].cpu().numpy())
        uv = torch.from_numpy(uv).cuda().float().unsqueeze(0) / 255.0
        # texture_img = imageio.imread("cache/uvassets/flower.png")
        # texture_img = torch.from_numpy(texture_img).cuda().float() / 255.0
        # texture_img = texture_img.permute(2, 0, 1).unsqueeze(0)
        texture_img = torch.zeros(1, 3, 128, 128, dtype=torch.float32, device=device)
        texture_img[:, 0, :, :] = 1.0
        texture_img[:, 1, :, :] = 1.0

        if args.renderer == "sh":
            # lightparam = torch.tensor([0.0, 1.0, 0.0], device=device)
            lightparam = torch.rand(9, device=device)

    img_target = torch.from_numpy(
        imageio.imread(DATA_DIR / "banana.png").astype(np.float32) / 255
    ).cuda()
    img_target = img_target[None, ...]  # .permute(0, 3, 1, 2)

    # Create a 'model' (an nn.Module) that wraps around the vertices, making it 'optimizable'.
    # TODO: Replace with a torch optimizer that takes vertices as a 'params' argument.
    # Deform the vertices slightly.
    model = Model(vertices.clone()).cuda()
    renderer.set_look_at_parameters([90 - azimuth], [elevation], [camera_distance])
    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
    mseloss = torch.nn.MSELoss()

    # Perform vertex optimization.
    if not args.no_viz:
        writer = imageio.get_writer(progressfile, mode="I")
    for i in trange(args.iters):
        optimizer.zero_grad()
        new_vertices = model()
        if args.renderer == "vc":
            img_pred, alpha, _ = renderer.forward(
                points=[new_vertices, faces[0].long()], colors_bxpx3=textures
            )
        elif args.renderer == "lam":
            img_pred, alpha, _ = renderer.forward(
                points=[new_vertices, faces[0].long()],
                uv_bxpx2=uv,
                texture_bx3xthxtw=texture_img,
            )
        elif args.renderer == "sh":
            img_pred, alpha, _ = renderer.forward(
                points=[new_vertices, faces[0].long()],
                uv_bxpx2=uv,
                texture_bx3xthxtw=texture_img,
                lightparam=lightparam,
            )
        rgba = torch.cat((img_pred, alpha), dim=-1)
        laplacian_loss = compute_laplacian_loss()
        loss = mseloss(rgba, img_target) + laplacian_loss
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            # TODO: Add functionality to write to gif output file.
            tqdm.write(f"Loss: {loss.item():.5}")
            if not args.no_viz:
                img = img_pred[0].detach().cpu().numpy()
                writer.append_data((255 * img).astype(np.uint8))
    if not args.no_viz:
        writer.close()

        # Write optimized mesh to output file.
        writer = imageio.get_writer(outfile, mode="I")
        for azimuth in trange(0, 360, 6):
            renderer.set_look_at_parameters(
                [90 - azimuth], [elevation], [camera_distance]
            )
            if args.renderer == "vc":
                img_pred, alpha, _ = renderer.forward(
                    points=[new_vertices, faces[0].long()], colors_bxpx3=textures
                )
            elif args.renderer == "lam":
                img_pred, alpha, _ = renderer.forward(
                    points=[new_vertices, faces[0].long()],
                    uv_bxpx2=uv,
                    texture_bx3xthxtw=texture_img,
                )
            elif args.renderer == "sh":
                img_pred, alpha, _ = renderer.forward(
                    points=[new_vertices, faces[0].long()],
                    uv_bxpx2=uv,
                    texture_bx3xthxtw=texture_img,
                    lightparam=lightparam,
                )
            img = img_pred[0].detach().cpu().numpy()
            writer.append_data((255 * img).astype(np.uint8))
        writer.close()

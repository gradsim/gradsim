# Example to sanity check whether SoftRas works.

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

from gradsim.renderutils import SoftRenderer, TriangleMesh

# Example script that uses SoftRas to optimize the texture for a given mesh.


class Model(torch.nn.Module):
    """Wrap textures into an nn.Module, for optimization. """

    def __init__(self, textures):
        super(Model, self).__init__()
        self.textures = torch.nn.Parameter(textures)

    def forward(self):
        return torch.sigmoid(self.textures)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of iterations to run optimization for.",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization steps."
    )
    args = parser.parse_args()

    # Initialize the soft rasterizer.
    renderer = SoftRenderer(camera_mode="look_at", device="cuda:0")

    # Camera settings.
    camera_distance = (
        2.0  # Distance of the camera from the origin (i.e., center of the object)
    )
    elevation = 30.0  # Angle of elevation
    azimuth = 0.0  # Azimuth angle

    # Directory in which sample data is located.
    DATA_DIR = Path(__file__).parent / "sampledata"

    # Read in the input mesh. TODO: Add filepath as argument.
    mesh = TriangleMesh.from_obj(DATA_DIR / "banana.obj")

    # Output filename to write out a rendered .gif to, showing the progress of optimization.
    progressfile = "texture_optimization_progress.gif"
    # Output filename to write out a rendered .gif file to, rendering the optimized mesh.
    outfile = "texture_optimization_output.gif"

    # Extract the vertices, faces, and texture the mesh (currently color with white).
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()
    textures = torch.ones(1, faces.shape[1], 2, 3, dtype=torch.float32, device="cuda:0")

    # Translate the mesh such that its centered at the origin.
    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.0
    vertices = vertices - vertices_middle
    # Scale the vertices slightly (so that they occupy a sizeable image area).
    # Skip if using models other than the banana.obj file.
    coef = 5
    vertices = vertices * coef

    img_target = torch.from_numpy(
        imageio.imread(DATA_DIR / "banana.png").astype(np.float32) / 255
    ).cuda()
    img_target = img_target[None, ...].permute(0, 3, 1, 2)

    # Create a 'model' (an nn.Module) that wraps around the vertices, making it 'optimizable'.
    # TODO: Replace with a torch optimizer that takes vertices as a 'params' argument.
    # Deform the vertices slightly.
    model = Model(textures).cuda()
    # renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
    optimizer = torch.optim.Adam(model.parameters(), 1.0, betas=(0.5, 0.99))
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)
    mseloss = torch.nn.MSELoss()

    # Perform texture optimization.
    if not args.no_viz:
        writer = imageio.get_writer(progressfile, mode="I")
    for i in trange(args.iters):
        optimizer.zero_grad()
        textures = model()
        rgba = renderer.forward(vertices, faces, textures)
        loss = mseloss(rgba, img_target)
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            # TODO: Add functionality to write to gif output file.
            tqdm.write(f"Loss: {loss.item():.5}")
            if not args.no_viz:
                img = rgba[0].permute(1, 2, 0).detach().cpu().numpy()
                writer.append_data((255 * img).astype(np.uint8))
    if not args.no_viz:
        writer.close()

        # Write optimized mesh to output file.
        writer = imageio.get_writer(outfile, mode="I")
        for azimuth in trange(0, 360, 6):
            renderer.set_eye_from_angles(camera_distance, elevation, azimuth)
            rgba = renderer.forward(vertices, faces, model())
            img = rgba[0].permute(1, 2, 0).detach().cpu().numpy()
            writer.append_data((255 * img).astype(np.uint8))
        writer.close()

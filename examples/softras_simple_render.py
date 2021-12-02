# Example to sanity check whether SoftRas works.

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import trange

from gradsim.renderutils import SoftRenderer, TriangleMesh

# Example script that uses SoftRas to render an image, given a mesh input

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stride",
        type=int,
        default=6,
        help="Rotation (in degrees) between successive render azimuth angles.",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization steps."
    )
    args = parser.parse_args()

    # Initialize the soft rasterizer.
    renderer = SoftRenderer(camera_mode="look_at", device="cuda:0")

    # Camera settings.
    camera_distance = (
        2.0  # Distance of the camera from the origin (i.e., center of the object).
    )
    elevation = 30.0  # Angle of elevation

    # Directory in which sample data is located.
    DATA_DIR = Path(__file__).parent / "sampledata"

    # Read in the input mesh.
    mesh = TriangleMesh.from_obj(DATA_DIR / "banana.obj")

    # Output filename (to write out a rendered .gif to).
    outfile = "cache/softras_render.gif"

    # Extract the vertices, faces, and texture the mesh (currently color with white).
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()
    # Initialize all faces to yellow (to color the banana)!
    textures = torch.cat(
        (
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device="cuda:0"),
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device="cuda:0"),
            torch.zeros(1, faces.shape[1], 2, 1, dtype=torch.float32, device="cuda:0"),
        ),
        dim=-1,
    )

    # Translate the mesh such that its centered at the origin.
    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.0
    vertices = vertices - vertices_middle
    # Scale the vertices slightly (so that they occupy a sizeable image area).
    # Skip if using models other than the banana.obj file.
    coef = 5
    vertices = vertices * coef

    # Loop over a set of azimuth angles, and render the image.
    print("Rendering using softras...")
    if not args.no_viz:
        writer = imageio.get_writer(outfile, mode="I")
    for azimuth in trange(0, 360, args.stride):
        renderer.set_eye_from_angles(camera_distance, elevation, azimuth)
        # Render an image.
        rgba = renderer.forward(vertices, faces, textures)
        if not args.no_viz:
            img = rgba[0].permute(1, 2, 0).detach().cpu().numpy()
            writer.append_data((255 * img).astype(np.uint8))
    if not args.no_viz:
        writer.close()

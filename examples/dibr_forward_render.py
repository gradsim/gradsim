# Adapted from Kaolin.

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
import tqdm
from PIL import Image

from gradsim.renderutils import TriangleMesh
from gradsim.renderutils.dibr.renderer import Renderer as DIBRenderer
from gradsim.renderutils.dibr.utils.sphericalcoord import get_spherical_coords_x


def main():

    ROOT_DIR = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description=" DIB-R Example")

    parser.add_argument(
        "--mesh",
        type=str,
        default=ROOT_DIR / "sampledata" / "banana.obj",
        help="Path to the mesh OBJ file",
    )
    parser.add_argument(
        "--use-texture", action="store_true", help="Whether to render a textured mesh."
    )
    parser.add_argument(
        "--texture",
        type=str,
        default=ROOT_DIR / "sampledata" / "texture.png",
        help="Specifies path to the texture to be used.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=ROOT_DIR / "cache" / "dibr",
        help="Output directory.",
    )

    args = parser.parse_args()

    CAMERA_DISTANCE = 2
    CAMERA_ELEVATION = 30
    MESH_SIZE = 5
    HEIGHT = 256
    WIDTH = 256

    mesh = TriangleMesh.from_obj(args.mesh)
    vertices = mesh.vertices.cuda()
    faces = mesh.faces.long().cuda()

    # Expand such that batch size = 1
    vertices = vertices.unsqueeze(0)

    ###########################
    # Normalize mesh position
    ###########################

    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.0
    vertices = (vertices - vertices_middle) * MESH_SIZE

    ###########################
    # Generate vertex color
    ###########################

    if not args.use_texture:
        vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
        vert_max = torch.max(vertices, dim=1, keepdims=True)[0]
        colors = (vertices - vert_min) / (vert_max - vert_min)

    ###########################
    # Generate texture mapping
    ###########################

    if args.use_texture:
        uv = get_spherical_coords_x(vertices[0].cpu().numpy())
        uv = torch.from_numpy(uv).cuda()

        # Expand such that batch size = 1
        uv = uv.unsqueeze(0)

    ###########################
    # Load texture
    ###########################

    if args.use_texture:
        # Load image as numpy array
        texture = np.array(Image.open(args.texture))

        # Convert numpy array to PyTorch tensor
        texture = torch.from_numpy(texture).cuda()

        # Convert from [0, 255] to [0, 1]
        texture = texture.float() / 255.0

        # Convert to NxCxHxW layout
        texture = texture.permute(2, 0, 1).unsqueeze(0)

    ###########################
    # Render
    ###########################

    if args.use_texture:
        renderer_mode = "Lambertian"

    else:
        renderer_mode = "VertexColor"

    renderer = DIBRenderer(HEIGHT, WIDTH, mode=renderer_mode)

    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description("Drawing")

    args.output_path.mkdir(exist_ok=True)
    savename = (
        "rendered_vertexcolor.gif" if not args.use_texture else "rendered_texture.gif"
    )
    writer = imageio.get_writer(args.output_path / savename, mode="I")
    for azimuth in loop:
        renderer.set_look_at_parameters(
            [90 - azimuth], [CAMERA_ELEVATION], [CAMERA_DISTANCE]
        )

        if args.use_texture:
            predictions, _, _ = renderer(
                points=[vertices, faces.long()], uv_bxpx2=uv, texture_bx3xthxtw=texture
            )

        else:
            predictions, _, _ = renderer(
                points=[vertices, faces.long()], colors_bxpx3=colors
            )

        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image * 255).astype(np.uint8))

    writer.close()


if __name__ == "__main__":
    main()

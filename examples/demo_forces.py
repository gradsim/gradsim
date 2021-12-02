from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import trange

from gradsim.bodies import RigidBody
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.simulator import Simulator
from gradsim.utils import meshutils

if __name__ == "__main__":

    # Device to store tensors on (MUST be CUDA-capable, for renderer to work).
    device = "cuda:0"

    # Output (gif) file path
    outfile = Path("cache/demoforces.gif")

    # Load a body (from a triangle mesh obj file).
    mesh = TriangleMesh.from_obj(Path("sampledata/cube.obj"))
    vertices = meshutils.normalize_vertices(mesh.vertices.unsqueeze(0)).to(device)
    faces = mesh.faces.to(device).unsqueeze(0)
    textures = torch.cat(
        (
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )
    masses = torch.nn.Parameter(
        0.1 * torch.ones(vertices.shape[1], dtype=vertices.dtype, device=device),
        requires_grad=True,
    )
    body = RigidBody(vertices[0], masses=masses)

    # Create a force that applies gravity (g = 10 metres / second^2).
    # gravity = Gravity(device=device)
    gravity = ConstantForce(direction=torch.tensor([0.0, -1.0, 0.0]), device=device)

    # Add this force to the body.
    body.add_external_force(gravity, application_points=[0, 1])

    # Initialize the simulator with the body at the origin.
    sim = Simulator([body])

    # Initialize the renderer.
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 8.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    # Run the simulation.
    writer = imageio.get_writer(outfile, mode="I")
    for i in trange(20):
        sim.step()
        # print("Body is at:", body.position)
        rgba = renderer.forward(body.get_world_vertices().unsqueeze(0), faces, textures)
        img = rgba[0].permute(1, 2, 0).detach().cpu().numpy()
        writer.append_data((255 * img).astype(np.uint8))
    writer.close()

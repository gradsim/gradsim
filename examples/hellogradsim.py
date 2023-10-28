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

    # Create cache if it does not previously exist
    cache = Path("cache")
    cache.mkdir(exist_ok=True)

    # Output (gif) file path
    outfile = cache / "hellogradsim.gif"

    # Load a body (from a triangle mesh obj file).
    mesh = TriangleMesh.from_obj(Path("sampledata/banana.obj"))
    vertices = meshutils.normalize_vertices(mesh.vertices.unsqueeze(0)).to(device)
    faces = mesh.faces.to(device).unsqueeze(0)
    textures = torch.ones(1, faces.shape[1], 2, 3, device=device)

    body = RigidBody(vertices[0])

    # Create a force that applies gravity (g = 10 metres / second^2).
    # gravity = Gravity(device=device)
    gravity = ConstantForce(direction=torch.tensor([0.0, -1.0, 0.0]), device=device)

    # Add this force to the body.
    body.add_external_force(gravity)

    # Initialize the simulator with the body at the origin.
    sim = Simulator([body])

    # Initialize the renderer.
    renderer = SoftRenderer(
        image_size=512,
        camera_mode="look_at", device=device)
    camera_distance = 8.0
    elevation = 30.0

    # Run the simulation.
    writer = imageio.get_writer(outfile, mode="I")
    num_steps = 20
    for i in trange(num_steps):
        # sim.step()
        # print("Body is at:", body.position)
        azimuth = 360 * i / num_steps
        renderer.set_eye_from_angles(camera_distance, elevation, azimuth)
        rgba = renderer.forward(body.get_world_vertices().unsqueeze(0), faces, textures)
        img = rgba[0].permute(1, 2, 0).detach().cpu().numpy()
        #     # import matplotlib.pyplot as plt
        #     # plt.imshow(rgba[0].permute(1, 2, 0).detach().cpu().numpy())
        #     # plt.show()
        writer.append_data((255 * img).astype(np.uint8))
    writer.close()

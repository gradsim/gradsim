import torch

from gradsim.bodies import RigidBody
from gradsim.forces import Gravity
from gradsim.simulator import Simulator


def test_smoke():
    sim = Simulator([])


def test_newtons_first_law_rest():
    # When no external force acts, bodies at rest remain at rest.
    cube_verts = torch.FloatTensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ]
    )
    cube = RigidBody(cube_verts)
    sim = Simulator([cube])
    sim.step()
    assert torch.allclose(cube.position, torch.zeros(3))
    sim.step()
    assert torch.allclose(cube.position, torch.zeros(3))


def test_newtons_first_law_motion():
    # When no external force acts, a body with constant velocity
    # continues to move with that velocity.
    cube_verts = torch.FloatTensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ]
    )
    cube = RigidBody(cube_verts)
    # Give the cube a linear momentum of `(8, 8, 8)` (it's mass is `8`),
    # so, the velocity becomes `1` meter per second. Our frame rate by
    # default, is 30 Hz, i.e., 0.033 meters per frame.
    cube.linear_momentum = torch.ones(3, dtype=torch.float32) * 8
    sim = Simulator([cube])
    sim.step()
    assert torch.allclose(cube.position, 0.0333 * torch.ones(3), atol=1e-4)
    sim.step()
    assert torch.allclose(cube.position, 0.0667 * torch.ones(3), atol=1e-4)
    sim.step()
    assert torch.allclose(cube.position, 0.1 * torch.ones(3), atol=1e-4)


def test_cube_with_gravity():
    # Add gravity to the cube.
    cube_verts = torch.FloatTensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ]
    )
    cube = RigidBody(cube_verts)
    gravity = Gravity()
    direction = torch.tensor([0.0, 0.0, -1.0])
    cube.add_external_force(gravity)
    sim = Simulator([cube])
    sim.step()
    assert torch.allclose(cube.linear_velocity, 0.3333333 * direction)
    assert torch.allclose(cube.position, 0.0111111 * direction)
    sim.step()
    assert torch.allclose(cube.linear_velocity, 0.66666667 * direction)
    assert torch.allclose(cube.position, 0.0333333 * direction)
    sim.step()
    assert torch.allclose(cube.linear_velocity, 1.0 * direction)
    assert torch.allclose(cube.position, 0.0666667 * direction)

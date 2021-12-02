import math

import torch

from gradsim.utils import quaternion


def test_normalize():
    quat = torch.randn(4)
    assert torch.allclose(quaternion.normalize(quat).norm(), torch.ones(1), atol=1e-4)


def test_quaternion_to_rotmat():
    # Create quaternion for rotation of pi radians about Y-axis; compare with rotmat.
    axis = torch.tensor([0.0, 1.0, 0.0])
    halfangle = torch.tensor([math.pi / 2])
    cos = torch.cos(halfangle)
    sin = torch.sin(halfangle)
    quat = torch.cat((cos, sin * axis), dim=0)
    rotmat = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0],])
    assert torch.allclose(quaternion.quaternion_to_rotmat(quat), rotmat, atol=1e-5)


def test_quaternion_multiply():
    # Create quat for rotation of pi about Y-axis. Create another quat for rotation
    # of -pi. Check if we get to identity.
    axis = torch.tensor([0.0, 1.0, 0.0])
    halfangle = torch.tensor([math.pi / 2])
    q1 = torch.cat((torch.cos(halfangle), torch.sin(halfangle) * axis), dim=0)
    q2 = torch.cat((torch.cos(-halfangle), torch.sin(-halfangle) * axis), dim=0)
    assert torch.allclose(
        quaternion.multiply(q1, q2), torch.tensor([1.0, 0.0, 0.0, 0.0])
    )

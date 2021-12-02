import pytest
import torch

from gradsim.bodies import RigidBody


def test_assertions():
    pytest.raises(TypeError, RigidBody)


def test_create_body():
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

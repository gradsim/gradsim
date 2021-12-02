import torch

from gradsim.forces import ConstantForce, Gravity, XForce, YForce


def test_constantforce():
    direction = torch.randn(3)
    magnitude = 10.0
    force = ConstantForce(direction, magnitude, starttime=0.5, endtime=1.0)
    assert torch.allclose(force.apply(0.1), torch.zeros(3))
    assert torch.allclose(force.apply(1.1), torch.zeros(3))
    assert torch.allclose(force.apply(0.5), direction * magnitude)
    assert torch.allclose(force.apply(0.9), direction * magnitude)


def test_gravity():
    direction = torch.tensor([0.0, 0.0, -1.0])
    force = Gravity()
    assert torch.allclose(force.apply(0.1), direction * 10.0)
    assert torch.allclose(force.apply(1.1), direction * 10.0)
    magnitude = 1.0
    force = Gravity(magnitude=magnitude)
    assert torch.allclose(force.apply(0.5), direction * magnitude)
    assert torch.allclose(force.apply(0.9), direction * magnitude)


def test_xforce():
    direction = torch.tensor([1.0, 0.0, 0.0])
    force = XForce()
    assert torch.allclose(force.apply(0.1), direction * 10.0)
    assert torch.allclose(force.apply(1.1), direction * 10.0)
    magnitude = 1.0
    force = XForce(magnitude=magnitude)
    assert torch.allclose(force.apply(0.5), direction * magnitude)
    assert torch.allclose(force.apply(0.9), direction * magnitude)


def test_yforce():
    direction = torch.tensor([0.0, 1.0, 0.0])
    force = YForce()
    assert torch.allclose(force.apply(0.1), direction * 10.0)
    assert torch.allclose(force.apply(1.1), direction * 10.0)
    magnitude = 1.0
    force = YForce(magnitude=magnitude)
    assert torch.allclose(force.apply(0.5), direction * magnitude)
    assert torch.allclose(force.apply(0.9), direction * magnitude)

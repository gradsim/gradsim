from abc import abstractmethod

import torch


class ExternalForce(object):
    """A generic external force to be applied to rigid bodies.

    Takes in a direction vector along which the force is applied, and the magnitude
    of the applied force. The `force()` method computes the force vector at the
    specified timestep.
    """

    def __init__(
        self,
        direction,
        magnitude=10.0,
        starttime=0.0,
        endtime=1e5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ):
        r"""Initialize an external force object with the specified direction and
        magnitude.

        Args:
            direction (torch.Tensor): Direction of the applied force
                (shape: :math:`(3)`).
            magnitude (float): Magnitude of the applied force (default: 100.0).
            starttime (float): Time (in seconds) at which the force is first applied
                (default: 0.0).
            endtime (float): Time (in seconds) at which the force stops being applied
                (default: 1e5).
            dtype (torch.dtype): `dtype` attribute of the `direction` variable.
            device (torch.device): `device` attribute of the `direction` variable.
        """
        self.direction = direction.to(dtype).to(device)
        self.magnitude = magnitude
        self.starttime = starttime
        self.endtime = endtime

    def apply(self, time):
        r"""Return the force vector at the time specified.

        Args:
            time (float): Time (in seconds).
        """
        if time < self.starttime or time > self.endtime:
            return self.direction * 0
        else:
            return self.force_function(time)

    @abstractmethod
    def force_function(self, time, *args, **kwargs):
        raise NotImplementedError


class ConstantForce(ExternalForce):
    """A constant force, with specified start and end times. """

    def __init__(
        self,
        direction,
        magnitude=10.0,
        starttime=0.0,
        endtime=1e5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ):
        super().__init__(direction, magnitude, starttime, endtime, dtype, device)

    def force_function(self, time):
        return self.direction * self.magnitude


class Gravity(ConstantForce):
    """A constant, downward force. """

    def __init__(
        self,
        direction=None,
        magnitude=10.0,
        starttime=0.0,
        endtime=1e5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ):
        if direction is None:
            direction = torch.tensor([0.0, 0.0, -1.0], dtype=dtype, device=device)
        super().__init__(direction, magnitude, starttime, endtime, dtype, device)


class XForce(ConstantForce):
    """A constant, downward force. """

    def __init__(
        self,
        direction=None,
        magnitude=10.0,
        starttime=0.0,
        endtime=1e5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ):
        if direction is None:
            direction = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        super().__init__(direction, magnitude, starttime, endtime, dtype, device)


class YForce(ConstantForce):
    """A constant, downward force. """

    def __init__(
        self,
        direction=None,
        magnitude=10.0,
        starttime=0.0,
        endtime=1e5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ):
        if direction is None:
            direction = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        super().__init__(direction, magnitude, starttime, endtime, dtype, device)

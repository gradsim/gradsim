import torch


def assert_tensor(var, varname):
    r"""Assert that the variable is of type torch.Tensor. """
    if not torch.is_tensor(var):
        raise TypeError(
            f"Expected {varname} of type torch.Tensor. Got {type(var)} instead."
        )

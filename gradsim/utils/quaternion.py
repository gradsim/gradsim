import torch


def normalize(quaternion):
    r"""Normalizes a quaternion to unit norm.

    Args:
        quaternion (torch.Tensor): Quaternion to normalize (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (torch.Tensor): Normalized quaternion (shape: :math:`(4)`).
    """
    norm = quaternion.norm(p=2, dim=0) + 1e-5
    return quaternion / norm


def quaternion_to_rotmat(quaternion):
    r"""Converts a quaternion to a :math:`3 \times 3` rotation matrix.

    Args:
        quaternion (torch.Tensor): Quaternion to convert (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (torch.Tensor): rotation matrix (shape: :math:`(3, 3)`).
    """
    r = quaternion[0]
    i = quaternion[1]
    j = quaternion[2]
    k = quaternion[3]
    rotmat = torch.zeros(3, 3, dtype=quaternion.dtype, device=quaternion.device)
    twoisq = 2 * i * i
    twojsq = 2 * j * j
    twoksq = 2 * k * k
    twoij = 2 * i * j
    twoik = 2 * i * k
    twojk = 2 * j * k
    twori = 2 * r * i
    tworj = 2 * r * j
    twork = 2 * r * k
    rotmat[0, 0] = 1 - twojsq - twoksq
    rotmat[0, 1] = twoij - twork
    rotmat[0, 2] = twoik + tworj
    rotmat[1, 0] = twoij + twork
    rotmat[1, 1] = 1 - twoisq - twoksq
    rotmat[1, 2] = twojk - twori
    rotmat[2, 0] = twoik - tworj
    rotmat[2, 1] = twojk + twori
    rotmat[2, 2] = 1 - twoisq - twojsq
    return rotmat


def multiply(q1, q2):
    r"""Multiply two quaternions `q1`, `q2`.

    Args:
        q1 (torch.Tensor): First quaternion (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).
        q2 (torch.Tensor): Second quaternion (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (torch.Tensor): Quaternion product (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).
    """
    r1 = q1[0]
    v1 = q1[1:]
    r2 = q2[0]
    v2 = q2[1:]
    return torch.cat(
        (
            r1 * r2 - torch.matmul(v1.view(1, 3), v2.view(3, 1)).view(-1),
            r1 * v2 + r2 * v1 + torch.cross(v1, v2),
        ),
        dim=0,
    )

import torch

from .utils.defaults import Defaults


def detect_ground_plane_contacts(vertices_world, eps=Defaults.EPSILON):
    """Detect contact points (vertices) with the ground-plane, given a set of
    vertices (usually --- belonging to a single mesh).

    Args:
        vertices_world (torch.Tensor): Set of vertices whose collisions with
            the ground plane (assumed to be the XZ-plane) are to be detected.
        eps (float): Contact detection threshold (i.e., distance below which
            two bodies will be considered penetrating).

    Returns:
        contact_inds (torch.Tensor): Indices of contact vertices.
        contact_points (torch.Tensor): Positions of contact vertices.
        contact_normals (torch.Tensor): Normals of contact (i.e., ground-plane
            normals here).
    """
    if eps < 0:
        raise ValueError(f"eps cannot be negative! Got: {eps}")
    contact_inds = torch.nonzero((vertices_world < eps)[..., 1]).view(-1)
    contact_points, contact_normals = None, None
    if contact_inds.numel() > 0:
        contact_points = vertices_world[contact_inds]
        contact_normals = torch.tensor(
            [0.0, 1.0, 0.0], dtype=vertices_world.dtype, device=vertices_world.device
        ).repeat(contact_inds.numel(), 1)
    else:
        contact_inds = None

    return contact_inds, contact_points, contact_normals

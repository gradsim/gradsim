def normalize_vertices(vertices, mean_subtraction=True, scale_factor=1.0):
    # vertices: N x 3
    if mean_subtraction:
        vertices = vertices - vertices.mean(-2).unsqueeze(-2)
    dists = vertices.norm(p=2, dim=-1)
    dist_max, _ = dists.max(dim=-1)
    vertices = scale_factor * (vertices / dist_max.unsqueeze(-1).unsqueeze(-1))
    return vertices

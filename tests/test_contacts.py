import torch

from gradsim.bodies import RigidBody
from gradsim.engines import (EulerIntegratorWithContacts,
                             SemiImplicitEulerWithContacts)
from gradsim.forces import ConstantForce
from gradsim.simulator import Simulator

if __name__ == "__main__":

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
    position = torch.tensor([2.0, 2.0, 2.0])
    orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cube = RigidBody(cube_verts + 1, position=position, orientation=orientation)
    force_magnitude = 10.0  # / cube_verts.shape[0]
    force_direction = torch.tensor([0.0, -1.0, 0.0])
    gravity = ConstantForce(magnitude=force_magnitude, direction=force_direction)
    cube.add_external_force(gravity)

    # sim = Simulator([cube], engine=EulerIntegratorWithContacts())

    sim_substeps = 32
    dtime = (1 / 30) / sim_substeps
    sim = Simulator([cube], engine=SemiImplicitEulerWithContacts(), dtime=dtime)

    # import numpy as np
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-1, 5)
    # ax.set_zlim(-2, 2)
    # plt.ion()

    print(cube.position)
    print("vertices at start:", cube.get_world_vertices())

    for i in range(800):
        sim.step()
        if i % sim_substeps == 0:
            print(cube.position)

    #     print("vertices")
    #     print(cube.get_world_vertices())

    #     # v = cube.get_world_vertices().detach().cpu().numpy()
    #     # plt.scatter(v[:, 0], v[:, 1], v[:, 2])
    #     # plt.show()
    #     # plt.pause(0.05)
    #     # plt.cla()

"""
A dflex smoketest
"""

from tqdm import trange

from gradsim import dflex

if __name__ == "__main__":

    # Initialize a `ModelBuilder` object to construct particle models.
    builder = dflex.sim.ModelBuilder()

    # Add an anchor point (set mass to zero).
    builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    # Build a chain (a spring-mass system).
    for i in range(1, 10):
        builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
        builder.add_spring(i - 1, i, 1.0e3, 0.0, 0)
    # Add a ground plane.
    builder.ground = True
    # Convert to a PyTorch simulation data structure.
    model = builder.finalize("cpu")
    # model = builder.finalize("cuda:0")

    # Simulation paramters
    sim_dt = 1.0 / 60.0
    sim_steps = 100

    # Integrator object
    integrator = dflex.sim.SemiImplicitIntegrator()

    # Initial state
    state = model.state()

    # Run (differentiable) simulation
    for i in trange(sim_steps):
        state = integrator.forward(model, state, sim_dt)
    print("Positions:", state.q)
    print("Velocities:", state.u)

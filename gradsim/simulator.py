from .contacts import detect_ground_plane_contacts
from .engines import EulerIntegrator
from .utils.defaults import Defaults


class Simulator(object):
    """A (differentiable) physics simulation world, with rigid bodies. """

    def __init__(
        self,
        bodies,
        dtime=Defaults.DT,
        engine=EulerIntegrator(),
        eps=Defaults.EPSILON,
        tol=Defaults.TOL,
        vrel_thresh=Defaults.VREL_THRESH,
        verbose=False,
        contacts=False,
    ):
        self.engine = engine
        self.time = 0.0
        self.dtime = dtime
        self.eps = eps
        self.tol = tol
        self.relative_velocity_threshold = vrel_thresh

        self.bodies = bodies

        if contacts:
            # Ensure no collisions occur initially
            contact_inds, _, _ = detect_ground_plane_contacts(
                bodies[0].get_world_vertices()
            )
            if contact_inds is not None:
                raise ValueError(
                    "Bad initial conditions. Contacts with ground-plane!"
                    f"Vertices: {contact_inds}."
                )

        self.verbose = verbose

    def step(self, dtime=None):
        if dtime is None:
            dtime = self.dtime
        # Integrator returns updated time (which is usually equal to dtime, but can be
        # much smaller, in case a collision is detected).
        dtime_update = self.engine.integrate(self, dtime)

        # for body, update in zip(self.bodies, state_updates):
        #     body.apply_state_update(update)
        self.time = self.time + dtime_update

    # def apply_external_forces(self):
    #     return [body.apply_external_forces(self.time) for body in bodies]

    def compute_state_derivatives(self):
        return [body.compute_state_derivatives(self.time) for body in self.bodies]

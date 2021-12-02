import torch

from .utils.asserts import assert_tensor
from .utils.defaults import Defaults
from .utils.quaternion import multiply, normalize, quaternion_to_rotmat


class RigidBody(object):
    r"""We define a rigid body in terms of a collection of "particles". By default,
    we assume a uniform mass distribution across the object, but variable-masses
    are possible.
    """

    def __init__(
        self,
        vertices,
        masses=None,
        position=None,
        orientation=None,
        linear_momentum=None,
        angular_momentum=None,
        friction_coefficient=Defaults.FRICTION_COEFFICIENT,
        restitution=Defaults.RESTITUTION,
        eps=Defaults.EPSILON,
        color=(255, 0, 0),
        thickness=1,
    ):
        assert_tensor(vertices, "vertices")
        self.dtype = vertices.dtype
        self.device = vertices.device

        # Initialize defaults wherever input params are None.
        if masses is None:
            masses = torch.ones(vertices.shape[0], dtype=self.dtype, device=self.device)
        if position is None:
            position = torch.zeros(3, dtype=self.dtype, device=self.device)
        if orientation is None:
            orientation = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=self.dtype, device=self.device
            )
        if linear_momentum is None:
            linear_momentum = torch.zeros(3, dtype=self.dtype, device=self.device)
        if angular_momentum is None:
            angular_momentum = torch.zeros(3, dtype=self.dtype, device=self.device)

        # Assert that desired params are wrapped as torch.Tensor objects.
        assert_tensor(masses, "masses")
        assert_tensor(position, "position")
        assert_tensor(orientation, "orientation")
        assert_tensor(linear_momentum, "linear_momentum")
        assert_tensor(angular_momentum, "angular_momentum")
        if vertices.dim() != 2 or vertices.shape[-1] != 3:
            raise ValueError(
                "vertices must have two dimensions, and the last dimension must"
                f" be of shape 3. Got shape {vertices.shape} instead."
            )

        # Translate vertices such that center-of-mass is at origin.
        com = self.compute_center_of_mass(vertices, masses)
        vertices = vertices - com.view(-1, 3)
        self.vertices = vertices
        self.masses = masses
        self.mass = masses.sum()  # Overall mass of the rigid body

        # State variables
        # self.position = self.compute_center_of_mass(self.vertices, self.masses)
        self.position = position
        self.orientation = orientation
        self.linear_momentum = linear_momentum
        self.angular_momentum = angular_momentum

        # Derived quantities

        # Body-frame inertia tensor.
        self.inertia_body = self.compute_inertia_body(self.vertices, self.masses)
        # Inverse of body-frame inertia tensor.
        self.inertia_body_inv = self.inertia_body.inverse()
        # Inverse of inertia tensor in world frame.
        self.inertia_world_inv = self.compute_inertia_world(
            self.inertia_body_inv, quaternion_to_rotmat(self.orientation),
        )
        self.linear_velocity = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.angular_velocity = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.force_vector = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.torque_vector = torch.zeros(3, dtype=self.dtype, device=self.device)

        # Friction coefficient
        self.friction_coefficient = None
        # Restitution
        self.restitution = restitution

        # List of external forces acting on the body.
        self.forces = []
        # List of points at which the forces are applied at (-1 indicates a force
        # applied to all points).
        self.application_points = []

        # Visualization / plotting parameters.
        self.color = color

        # Create object geometry, for collision testing with ODE.
        self._create_geometry()

    def _create_geometry(self):
        pass

    @staticmethod
    def compute_inertia_body(vertices, masses):
        r"""Compute the inertia tensor of the rigid-body, in the body frame.

        Recall that, for a rigid-body with N particles, each of mass :math:`m`,
        with position :math:`r_i` relative to the body frame (i.e., relative to
        the center of mass of the body), the inertia tensor in the body frame is
        given by :math:`I(t) = \sum_{i=1}^{N} (r_i^T r_i) \mathbf{1}_3 - r_i r_i^T`.

        Args:
            vertices (torch.Tensor): Vertices of the rigid-body (shape: :math:`(N, 3)`)
            masses (torch.Tensor): Mass of each particle (shape: :math:`(N)`)

        Returns:
            (torch.Tensor): Inertia tensor, in the body frame (shape: :math:`(3, 3)`)
        """
        N = vertices.shape[0]  # Number of vertices
        # rt_r: (N, 1, 1)
        rt_r = torch.matmul(vertices.view(-1, 1, 3), vertices.view(-1, 3, 1))
        # r_rt: (N, 3, 3)
        r_rt = torch.matmul(vertices.view(-1, 3, 1), vertices.view(-1, 1, 3))
        eye = torch.eye(3, dtype=vertices.dtype, device=vertices.device)
        # Compute and return the (3, 3) inertia tensor.
        return ((rt_r * eye.repeat(N, 1, 1) - r_rt) * masses.view(N, 1, 1)).sum(0)

    @staticmethod
    def compute_inertia_world(inertia_body_inv, rotmat):
        r"""Compute the inertia tensor of the rigid-body, in the world frame, as
        specified by the rotation matrix `rotmat`.

        Note that the translation is not needed, as inertia is translation-invariant.
        Given the body-frame inertia tensor :math:`I_{body}`, its counterpart in the
        world-frame (specified by the :math:`3 \times 3` rotation matrix :math:`R`)
        is given by :math:`I_{world} = R I_{body}^{-1} R^T`.

        Args:
            inertia_body_inv (torch.Tensor): Body-frame inertia tensor inversed
                (shape: :math:`(3, 3`).
            rotmat (torch.Tensor): Rotation matrix that specifies the body-frame wrt
                the world-frame (shape: :math:`(3, 3)`).

        Returns:
            (torch.Tensor): World-frame inertia tensor (shape: :math:`(3, 3`).
        """
        return torch.matmul(
            torch.matmul(rotmat, inertia_body_inv), rotmat.transpose(0, 1)
        )

    @staticmethod
    def compute_angular_velocity_from_angular_momentum(
        inertia_world_inv, angular_momentum,
    ):
        r"""Computes angular velocity, given the inverse of the world-frame inertia
        tensor and the angular momentum vector.

        Args:
            inertia_world_inv (torch.Tensor): Inverse of the world-frame inertia
                tensor (shape: :math:`(3, 3)`).
            angular_momentum (torch.Tensor): Angular momentum (shape: :math:`(3)`).

        Returns:
            (torch.Tensor): Angular velocity (shape: :math:`(3)`).
        """
        return torch.matmul(inertia_world_inv, angular_momentum.view(-1, 1)).squeeze(-1)

    @staticmethod
    def compute_linear_velocity_from_linear_momentum(
        linear_momentum, mass,
    ):
        r"""Computes linear velocity, given linear momentum.

        Args:
            linear_momentum (torch.Tensor): Linear momentum (shape: :math:`(3)`).
            mass (torch.Tensor): Total mass (shape: :math:`(1)`).

        Returns:
            (torch.Tensor): Linear velocity (shape: :math:`(3)`).
        """
        return linear_momentum / mass

    @staticmethod
    def compute_center_of_mass(vertices, masses):
        r"""Computes the center of mass of the rigid-body (in the world-frame).

        Recall that, for a rigid-body with :math:`N` particles, the center of mass
        is given by :math:`\frac{1}{M} \sum_{i=1}^{N} m_i r_i`, where :math:`r_i`
        is the position of particle :math:`i` in the world-frame.

        Args:
            vertices (torch.Tensor): Vertices (positions of each particle) of the
                rigid-body (shape: :math:`(N, 3)`).
            masses (torch.Tensor): Mass of each particle (shape: :math:`(N)`).

        Returns:
            (torch.Tensor): Center of mass of the rigid-body (in the world-frame)
                (shape: :math:`(3)`).
        """
        return (masses.view(-1, 1) * vertices).sum(0) / masses.sum()

    def add_external_force(self, force, application_points=None):
        """Add an external force to the set of forces acting on the body.

        Args:
            force (gradsim.forces.ExternalForce): An external force object that
                has an `apply(time)` method that provides the instantaneous force
                vector acting on the body.
            application_points (iterable): Indices of particles (vertices) of the
                body on which the force applies. (Default: `None`; indicates that the
                force will be applied to all points on the body (and consequently
                produce pure translation)).
        """
        self.forces.append(force)
        self.application_points.append(application_points)

    def apply_external_forces(self, time):
        """Apply the external forces (includes torques) at the current timestep. """
        force_per_point = torch.zeros_like(self.vertices)
        torque_per_point = torch.zeros_like(self.vertices)

        for force, application_points in zip(self.forces, self.application_points):
            # Compute the force vector.
            force_vector = force.apply(time)
            torque = torch.zeros(3, dtype=self.dtype, device=self.device)
            if application_points is not None:
                mask = torch.zeros_like(self.vertices)
                inds = (
                    torch.tensor(
                        application_points, dtype=torch.long, device=self.device
                    )
                    .view(-1, 1)
                    .repeat(1, 3)
                )
                mask = mask.scatter_(0, inds, 1.0)
                force_per_point = force_per_point + mask * force_vector.view(1, 3)
                torque_per_point = torque_per_point + torch.cross(
                    self.vertices - self.position.view(1, 3), force_per_point
                )
            else:
                force_per_point = force_per_point + force_vector.view(1, 3)
                # Torque is 0 this case; axis of force passes through center of mass.

        return force_per_point.sum(0), torque_per_point.sum(0)

    def compute_linear_momentum(self, time):
        r"""Compute the linear momentum of the rigid-body at the current timesetp.
        """
        pass

    def compute_state_derivatives(self, time):
        r"""Compute the time-derivatives of the state vector, adopting the convention
        from Witkin and Baraff's SIGGRAPH '97 course.
        http://www.cs.cmu.edu/~baraff/sigcourse/index.html
        """

        # Derivative of position :math:`x(t)` is velocity :math:`v(t)`.
        # See Eq. (2-43) from above source.
        dposition = self.linear_velocity
        # Derivative of orientation :math:`q(t)` (quaternion representation) is
        # :math:`0.5 \omega(t) \circle q(t)`, where :math:`\circle` denotes
        # quaternion multiplication, where :math:`\omega(t)` is the angular velocity
        # converted to a quaternion (with `0` as the scalar component).
        # See Eq. (4-2) from above source.
        angular_velocity_quat = torch.zeros(4, dtype=self.dtype, device=self.device)
        angular_velocity_quat[1:] = self.angular_velocity
        dorientation = 0.5 * multiply(angular_velocity_quat, self.orientation)
        # Derivative of linear momentum :math:`P(t)` is force :math:`F(t)`.
        # See Eq. (2-43) from above source.
        # Derivative of angular momentum :math:`L(t)` is torque :math:`\tau(t)`.
        # See Eq. (2-43) from above source.
        dlinear_momentum, dangular_momentum = self.apply_external_forces(time)

        return dposition, dorientation, dlinear_momentum, dangular_momentum

    def finish_state_update(self):
        """Finish performing the state update, using integrated state estimates.

        Concretely, compute the linear and angular velocities, and normalize the
        orientation (quaternion).
        """
        self.orientation = normalize(self.orientation)
        self.linear_velocity = self.linear_momentum / self.masses.sum()
        inertia_world_inv = self.compute_inertia_world(
            self.inertia_body_inv, quaternion_to_rotmat(self.orientation)
        )
        self.angular_velocity = torch.matmul(
            inertia_world_inv, self.angular_momentum.view(-1, 1)
        ).view(-1)

    def get_world_vertices(self):
        """Returns vertices transformed to world-frame. """
        rotmat = quaternion_to_rotmat(self.orientation)
        return torch.matmul(rotmat, self.vertices.transpose(-1, -2)).transpose(
            -1, -2
        ) + self.position.view(1, 3)


class SimplePendulum(torch.nn.Module):
    r"""A simple pendulum object.

    The mathematical model for the pendulum object is given by the following
    ordinary differential equation (ODE).

    See https://skill-lync.com/projects/Simulation-of-a-Simple-Pendulum-on-Python-95518

    :math:`\ddot{\theta} + \frac{b}{m} \dot{\theta} + \frac{g}{L} \sin(\theta) = 0`

    Here, :math:`\ddot{\theta}` is the angular acceleration, :math:`\dot{\theta}` is
    the angular velocity, :math:`\theta` is the angular displacement, :math:`b` is the
    damping factor, :math:`g` is the acceleration due to gravity, :math:`L` is the
    length of the pendulum, and :math:`m` is the mass of the bob.

    This second order ODE can be converted to two first order ODEs. If you let
    :math:`\theta_1 = \theta` and :math:`\theta_2 = \dot{\theta} = \dot{\theta}_1`,
    the first order ODEs are as follows:

    :math:`\dot{\theta}_2 = -\frac{b}{m} \theta_2 - \frac{g}{L} \sin{\theta_1}`
    :math:`\dot{\theta}_1 = \theta_2`

    This can be solved by differentiable ODE routines (eg. `torchdifeq`), as opposed to
    explicit Euler integration.
    """

    def __init__(
        self, mass=None, radius=None, gravity=None, length=None, damping=None,
    ):
        r"""Initialize a simple pendulum object, with the parameters specified.

        Args:
            mass (float or torch.Tensor): Mass of the bob (kg) (assumed to be a point
                mass, i.e., a spherical bob).
            radius (float or torch.Tensor): Radius of the sphere (m) (currently only
                used for rendering and shape estimation).
            gravity (float or torch.Tensor): Acceleration due to gravity (m / s^2).
            length (float or torch.Tensor): Length of the pendulum (m) (distance
                between the point of suspension and the center-of-mass of the bob).
            damping (float or torch.Tensor): Damping coefficient (a dimensionless
                quantity).
        """
        super(SimplePendulum, self).__init__()
        if mass is None:
            mass = 1.0
        if radius is None:
            radius = 1.0
        if gravity is None:
            gravity = 10.0
        if length is None:
            length = 1.0
        if damping is None:
            damping = 0.5
        self.mass = mass
        self.radius = radius
        self.gravity = gravity
        self.length = length
        self.damping = damping

    def forward(self, time, theta):
        """Compute the simple pendulum ODE at the specified `time` values, given the
        initial conditions `theta`.
        """
        dtheta = torch.zeros(2, dtype=theta.dtype, device=theta.device)
        dtheta[0] = theta[1]
        dtheta[1] = (
            -(self.damping / self.mass) * theta[1]
            - (self.gravity / self.length) * theta[0].sin()
        )
        return dtheta


class DoublePendulum(torch.nn.Module):
    r"""A double pendulum object.

    As with the simple pendulum, we express the evolution of the double pendulum's
    state as an ordinary differential equation (ODE).

    The ODE is derived quite easily---while the math looks a tad cumbersome---by
    application of the law of conservation of energy. For a reference derivation,
    see: http://scienceworld.wolfram.com/physics/DoublePendulum.html
    This ODE is equivalent to computing the derivatives of the Hamiltonian of the
    system, wrt state parameters.

    For a reference Python implementation, see:
    https://scipython.com/blog/the-double-pendulum/
    """

    def __init__(self, length1=1.0, length2=1.0, mass1=1.0, mass2=1.0, gravity=10.0):
        """Initialize the double pendulum.

        Args:
            length1 (float or torch.Tensor): Length of the first rod/string (m).
            length2 (float or torch.Tensor): Length of the second rod/string (m).
            mass1 (float or torch.Tensor): Mass of the first bob (kg).
            mass2 (float or torch.Tensor): Mass of the second bob (kg).
            gravity (float or torch.Tensor): Acceleration due to gravity (m/s^2).
        """
        super(DoublePendulum, self).__init__()
        self.length1 = length1
        self.length2 = length2
        self.mass1 = mass1
        self.mass2 = mass2
        self.gravity = gravity

    def forward(self, time, y):
        """Computes time derivatives of the double pendulum ODE wrt state.

        NOTE: This module is intended for use with an ODE integrator, as opposed
        for directly computing state evolution! For example, this can be used with
        the `odeint` routine supplied by neural ode packages such as `torchdiffeq`.

        Args:
            time (torch.Tensor): Timesteps at which the ODE derivatives are to be
                evaluated (usually a 1-D tensor).
            y (torch.Tensor): State variables (state y = (theta1, z1, theta2, z2),
                where theta1 and theta2 are the angular displacements of the first
                and the second bobs respectively, and z1 and z2 are the angular
                velocities of the first and the second bobs.)

        Returns:
            dydt (torch.Tensor): State derivatives.
        """
        dydt = torch.zeros(4, dtype=y.dtype, device=y.device)

        theta1 = y[0]
        z1 = y[1]
        theta2 = y[2]
        z2 = y[3]
        c, s = (theta1 - theta2).cos(), (theta1 - theta2).sin()
        theta1dot = z1
        theta2dot = z2
        z1sq = z1 ** 2
        z2sq = z2 ** 2
        denominator = self.mass1 + self.mass2 * s ** 2

        # Helper variables to compute the various terms in the ODE
        term1 = self.mass2 * self.gravity * theta2.sin() * c
        term2 = self.mass2 * s * (self.length1 * z1sq * c + self.length2 * z2sq)
        term3 = (self.mass1 + self.mass2) * self.gravity * theta1.sin()
        z1dot = (term1 - term2 - term3) / (self.length1 * denominator)
        term4 = self.length1 * z1sq * s - self.gravity * theta2.sin()
        term5 = self.gravity * theta1.sin() * c
        term6 = self.mass2 * self.length2 * z2sq * s * c
        z2dot = ((self.mass1 + self.mass2) * (term4 + term5) + term6) / (
            self.length2 * denominator
        )

        dydt[0] = theta1dot
        dydt[1] = z1dot
        dydt[2] = theta2dot
        dydt[3] = z2dot

        return dydt

    def compute_energy(self, y):
        """Return the total energy of the system.

        Args:
            y (torch.Tensor): State at which to evaluate the total energy.
                State variables (state y = (theta1, z1, theta2, z2),
                where theta1 and theta2 are the angular displacements of the first
                and the second bobs respectively, and z1 and z2 are the angular
                velocities of the first and the second bobs.)
        """
        if y.ndim == 2:
            t1, t1d, t2, t2d = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        elif y.ndim == 1:
            t1, t1d, t2, t2d = y[0], y[1], y[2], y[3]
        else:
            raise ValueError(f"Input tensor must have 1 or 2 dims. Got {y.ndim} dims.")
        m1, m2, l1, l2 = self.mass1, self.mass2, self.length1, self.length2
        g = self.gravity
        V = -(m1 + m2) * l1 * g * t1.cos() - m2 * l2 * g * t2.cos()
        T = 0.5 * m1 * (l1 * t1d) ** 2 + 0.5 * m2 * (
            (l1 * t1d) ** 2
            + (l2 * t2d) ** 2
            + 2 * l1 * l2 * t1d * t2d * (t1 - t2).cos()
        )
        return T + V

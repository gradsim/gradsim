import math
from abc import ABCMeta, abstractmethod

import torch

from .contacts import detect_ground_plane_contacts
from .utils.quaternion import normalize, quaternion_to_rotmat


class ODEIntegrator(metaclass=ABCMeta):
    """Abstract base class for ODE integrators. Integrators have a
    `integrate()` method that computes time derivatives of the
    state vector.
    """

    @abstractmethod
    def integrate(self, *args, **kwargs):
        pass


class EulerIntegrator(ODEIntegrator):
    """Performs semi-implicit Euler integration to solve the ODE. """

    def __init__(self):
        super().__init__()

    def integrate(self, simulator, dtime):
        # Compute forces (and torques).
        derivatives = simulator.compute_state_derivatives()
        # Compute state updates.
        for body, dstate in zip(simulator.bodies, derivatives):
            # body.position = body.position + dstate[0] * dtime
            body.orientation = body.orientation + dstate[1] * dtime
            body.linear_momentum = body.linear_momentum + dstate[2] * dtime
            body.angular_momentum = body.angular_momentum + dstate[3] * dtime
            body.orientation = normalize(body.orientation)
            body.linear_velocity = body.linear_momentum / body.masses.sum()
            inertia_world_inv = body.compute_inertia_world(
                body.inertia_body_inv, quaternion_to_rotmat(body.orientation)
            )
            body.angular_velocity = torch.matmul(
                inertia_world_inv, body.angular_momentum.view(-1, 1)
            ).view(-1)
            # Update the position in the end, as that's when linear velocity is
            # available.
            body.position = body.position + body.linear_velocity * dtime

        return dtime


class EulerIntegratorWithContacts(ODEIntegrator):
    """Performs semi-implicit Euler integration with ground-plane contact detection, to
    solve the ODE.
    """

    def __init__(self):
        super().__init__()

    def get_world_vertices(self, vertices, quaternion, position):
        """Returns vertices transformed to world-frame. """
        rotmat = quaternion_to_rotmat(quaternion)
        return torch.matmul(rotmat, vertices.transpose(-1, -2)).transpose(
            -1, -2
        ) + position.view(1, 3)

    def compute_lookahead_state_update(self, starttime, endtime, body):
        if endtime < starttime:
            raise ValueError("This should never happen!")
        dtime_ = endtime - starttime
        dstate = body.compute_state_derivatives(starttime)

        orientation = body.orientation + dstate[1] * dtime_
        linear_momentum = body.linear_momentum + dstate[2] * dtime_
        angular_momentum = body.angular_momentum + dstate[3] * dtime_
        orientation = normalize(orientation)
        linear_velocity = linear_momentum / body.masses.sum()
        inertia_world_inv = body.compute_inertia_world(
            body.inertia_body_inv, quaternion_to_rotmat(orientation)
        )
        angular_velocity = torch.matmul(
            inertia_world_inv, angular_momentum.view(-1, 1)
        ).view(-1)
        # Update the position in the end, as that's when linear velocity is
        # available (important for semi-implicit Euler integration).
        position = body.position + linear_velocity * dtime_

        return position, orientation, linear_momentum, angular_momentum

    def detect_collision_in_time_window(self, starttime, endtime, body):
        # Compute a "lookahead" state (position -> pos, orientation -> rot,
        # linear momentum -> p, angular momentul -> L).
        pos, rot, p, L = self.compute_lookahead_state_update(starttime, endtime, body)
        # Detect collisions
        contact_inds, _, _ = detect_ground_plane_contacts(
            self.get_world_vertices(body.vertices, rot, pos)
        )
        return contact_inds is not None

    def bisection_method_for_toi(self, starttime, endtime, body):
        """Apply the bisection method and determine time-of-impact. """
        if endtime < starttime:
            raise ValueError("This should never happen!")
        if endtime - starttime <= 0.001:  # TODO: Replace with tolerance
            print("bisection termination:", endtime - starttime)
            return endtime
        midtime = 0.5 * (endtime - starttime)
        colliding_in_first_half = self.detect_collision_in_time_window(
            starttime, starttime + midtime, body
        )
        if colliding_in_first_half:
            return self.bisection_method_for_toi(starttime, starttime + midtime, body)
        else:
            return self.bisection_method_for_toi(starttime + midtime, endtime, body)
        raise ValueError("bisection should never reach here!")
        # print("########################")
        # print("colliding_in_first_half:", colliding_in_first_half)
        # colliding_in_second_half = self.detect_collision_in_time_window(
        #     starttime + midtime, endtime, body
        # )
        # print("colliding_in_second_half:", colliding_in_second_half)
        # if self.detect_collision_in_time_window(starttime, starttime + midtime, body):
        #     return self.bisection_method_for_toi(starttime, starttime + midtime, body)
        # elif self.detect_collision_in_time_window(starttime + midtime, endtime, body):
        #     return self.bisection_method_for_toi(starttime + midtime, endtime, body)
        # else:
        #     raise ValueError("bisection should never reach here!")

    def integrate(self, simulator, dtime):

        # WARNING: This will fail if the simulator contains multiple bodies.
        # NOTE: The ground plane (or other colliding planes) are not treated as
        # additional bodies.

        # Determine if a collision would occur if we rollout the current timestep.
        is_colliding_in_this_timestep = self.detect_collision_in_time_window(
            simulator.time, simulator.time + dtime, simulator.bodies[0]
        )

        # If lookahead state is colliding, determine toi (time-of-impact).
        toi = None
        if is_colliding_in_this_timestep:
            print("Calling bisection")
            toi = self.bisection_method_for_toi(
                simulator.time, simulator.time + dtime, simulator.bodies[0]
            )
            print("toi:", toi)

        if toi is not None:
            dtime = toi - simulator.time
            print("dtime changed to:", dtime)

        # Compute forces (and torques).
        derivatives = simulator.compute_state_derivatives()
        # Compute state updates.
        for body, dstate in zip(simulator.bodies, derivatives):
            # body.position = body.position + dstate[0] * dtime
            body.orientation = body.orientation + dstate[1] * dtime
            body.linear_momentum = body.linear_momentum + dstate[2] * dtime
            body.angular_momentum = body.angular_momentum + dstate[3] * dtime
            body.orientation = normalize(body.orientation)
            body.linear_velocity = body.linear_momentum / body.masses.sum()
            inertia_world_inv = body.compute_inertia_world(
                body.inertia_body_inv, quaternion_to_rotmat(body.orientation)
            )
            body.angular_velocity = torch.matmul(
                inertia_world_inv, body.angular_momentum.view(-1, 1)
            ).view(-1)
            # Update the position in the end, as that's when linear velocity is
            # available.
            body.position = body.position + body.linear_velocity * dtime

        # Handle contact events (apply a corrective impluse)!
        if toi is not None:
            (
                contact_inds,
                contact_points,
                contact_normals,
            ) = detect_ground_plane_contacts(body.get_world_vertices())
            if contact_inds is None:
                raise ValueError(
                    "An error has occured! Execution reached here because a contact "
                    "event was discovered in the first place, and a toi fix has been "
                    "applied, and now an impulse response needs to be computed."
                )
            else:
                print("Apply corrective impulse here!")
                # Since we are colliding with the ground plane, the relative velocity
                # is given by dot_product(groundplane_normal, vel_object - vel_ground).
                # We assume the ground plane (outwards) normal to be (0, 1, 0) and the
                # velocity of the ground plane to be (0, 0, 0). Further, we assume that
                # the mass of the ground plane is very large (so that its inverse mass
                # is zero). We follow equation (8-18) on page 17 of Witkin and Baraff's
                # SIGGRAPH 1997 course notes on "Physically based modeling: principles
                # and practice" (Rigid body dynamics - Lecture Notes II (motion with
                # constraints)) (http://www.cs.cmu.edu/~baraff/sigcourse/notesd2.pdf).

                # Get the velocity of contact vertices (linear velocity +
                # cross_product(angular velocity, contact_vertex_in_world_frame -
                # center_of_mass_in_world_frame)). (Eq. (8-1) Page 11 of above notes).

                for idx, val in enumerate(contact_inds):
                    r = contact_points[idx] - body.position
                    n = contact_normals[idx]
                    vrel = body.linear_velocity + torch.cross(body.angular_velocity, r)
                    vrel = torch.dot(n, vrel)

                    THRESHOLD = 0.001
                    if vrel > THRESHOLD:
                        continue
                    if vrel > -THRESHOLD:
                        continue

                    num = -(1 + body.restitution) * vrel
                    minv = 1 / (body.masses.sum())
                    iinv = body.compute_inertia_world(
                        body.inertia_body_inv, quaternion_to_rotmat(body.orientation)
                    )
                    term0 = torch.cross(r, n)
                    term1 = torch.matmul(iinv, term0.unsqueeze(-1)).squeeze(-1)
                    term2 = torch.cross(term1, r)
                    term3 = torch.dot(n, term2)
                    den = minv + term3
                    j = (num / den) * n

                    body.linear_momentum = body.linear_momentum + j
                    body.angular_momentum = body.angular_momentum + torch.cross(r, j)
                    body.linear_velocity = body.linear_momentum / body.masses.sum()
                    body.angular_velocity = torch.matmul(
                        inertia_world_inv, body.angular_momentum.view(-1, 1)
                    ).view(-1)

                return dtime

                """
                Attempt to do batch mode collision resolution (didn't work).
                """

                # # positions relative to center-of-mass
                # r = contact_points.view(-1, 3) - body.position.view(-1, 3)
                # contact_velocities = body.linear_velocity.view(-1, 3) + torch.cross(
                #     body.angular_velocity.view(-1, 3).repeat(r.shape[0], 1), r
                # )
                # # Apply dot_product(ground plane normal, contact_velocities).
                # # (Since it's a batch dot operation, implement using matmul).
                # # (N x 1 x 3) x (N x 3 x 1) -> (N x 1 x 1)
                # # Squeeze twice => (N x 1 x 1) -> (N,)
                # # print(contact_normals.shape, contact_velocities.shape)
                # contact_velocities = torch.matmul(
                #     contact_normals.unsqueeze(-2), contact_velocities.unsqueeze(-1)
                # ).squeeze(-1).squeeze(-1)
                # # print(contact_velocities.shape)

                # numerator = -(1 + 1) * contact_velocities
                # inv_mass = (1 / body.masses.sum())
                # inertia_world_inv = body.compute_inertia_world(
                #     body.inertia_body_inv, quaternion_to_rotmat(body.orientation)
                # )
                # # print("###")
                # # print(inertia_world_inv.shape)
                # # print(torch.cross(torch.cross(r, contact_normals), r).shape)
                # # print("###")
                # term0 = torch.cross(r, contact_normals)
                # term1 = torch.matmul(
                #     inertia_world_inv.unsqueeze(0),
                #     term0.unsqueeze(-1),
                # ).squeeze(-1)
                # # print(term0.shape, term1.shape)
                # term2 = torch.cross(term1, r)
                # # print(term2.shape)
                # term3 = torch.matmul(
                #     contact_normals.unsqueeze(-2), term2.unsqueeze(-1)
                # ).squeeze(-1).squeeze(-1)
                # # print(term3.shape)
                # denominator = inv_mass + term3

                # impulses = (numerator / (denominator)).unsqueeze(-1) * contact_normals
                # print("jn:", impulses.shape)

                # # for i in range(impulses.shape[0]):
                # #     body.linear_momentum = body.linear_momentum + impulses[i]
                # #     body.angular_momentum = body.angular_momentum + torch.cross(r[i], impulses[i])
                # #     body.linear_velocity = body.linear_momentum / body.masses.sum()
                # #     body.angular_velocity = torch.matmul(
                # #         inertia_world_inv, body.angular_momentum.view(-1, 1)
                # #     ).view(-1)

                # # linear_momentum += impulse_forces
                # body.linear_momentum = body.linear_momentum + impulses.sum(0)
                # # angular_momentum += sum(cross(r[i], impulse_force[i]))
                # body.angular_momentum = body.angular_momentum + torch.cross(r, impulses).sum(0)
                # # linear_velocity = linear_momentum / mass
                # body.linear_velocity = body.linear_momentum / body.masses.sum()
                # # angular_velocity = inertia_world_inv * angular_momentum
                # body.angular_velocity = torch.matmul(
                #     inertia_world_inv, body.angular_momentum.view(-1, 1)
                # ).view(-1)

                """
                Another attempt at batch-model collision resolution (didn't work!)
                """

                # r = contact_points.view(-1, 3) - body.position.view(-1, 3)
                # n = contact_normals.view(-1, 3)
                # vrel = body.linear_velocity.view(-1, 3) + torch.cross(
                #     body.angular_velocity.view(-1, 3).repeat(r.shape[0], 1), r
                # )
                # vrel = torch.matmul(
                #     n.unsqueeze(-2), vrel.unsqueeze(-1)
                # ).squeeze(-1)
                # num = -(1 + body.restitution) * vrel
                # minv = 1 / body.masses.sum()
                # iinv = body.compute_inertia_world(
                #     body.inertia_body_inv, quaternion_to_rotmat(body.orientation)
                # )
                # term0 = torch.cross(r, n)
                # term1 = torch.matmul(
                #     iinv.unsqueeze(0).repeat(r.shape[0], 1, 1), term0.unsqueeze(-1)
                # ).squeeze(-1)
                # term2 = torch.cross(term1, r)
                # term3 = torch.matmul(
                #     n.unsqueeze(-2), term2.unsqueeze(-1)
                # ).squeeze(-1)
                # den = minv + term3
                # j = (num / den) * n
                # print(j.shape)

                # body.linear_momentum = body.linear_momentum + j.sum(0)
                # body.angular_momentum = body.angular_momentum + torch.cross(r, j).sum(0)
                # body.linear_velocity = body.linear_momentum / body.masses.sum()
                # body.angular_velocity = torch.matmul(
                #     inertia_world_inv, body.angular_momentum.view(-1, 1)
                # ).view(-1)

                # return dtime

        # contact_inds, _, _ = detect_ground_plane_contacts(body.get_world_vertices())
        # if contact_inds is not None:
        #     print("Contact event!")
        #     # raise ValueError("Contact event!")

        return dtime


class SemiImplicitEulerWithContacts(ODEIntegrator):
    """Performs semi-implicit Euler integration with ground-plane contact detection, to
    solve the ODE.
    """

    def __init__(self):
        super().__init__()

    def get_world_vertices(self, vertices, quaternion, position):
        """Returns vertices transformed to world-frame. """
        rotmat = quaternion_to_rotmat(quaternion)
        return torch.matmul(rotmat, vertices.transpose(-1, -2)).transpose(
            -1, -2
        ) + position.view(1, 3)

    def compute_lookahead_state_update(self, starttime, endtime, body):
        if endtime < starttime:
            raise ValueError("This should never happen!")
        dtime_ = endtime - starttime
        dstate = body.compute_state_derivatives(starttime)

        orientation = body.orientation + dstate[1] * dtime_
        linear_momentum = body.linear_momentum + dstate[2] * dtime_
        angular_momentum = body.angular_momentum + dstate[3] * dtime_
        orientation = normalize(orientation)
        linear_velocity = linear_momentum / body.masses.sum()
        inertia_world_inv = body.compute_inertia_world(
            body.inertia_body_inv, quaternion_to_rotmat(orientation)
        )
        angular_velocity = torch.matmul(
            inertia_world_inv, angular_momentum.view(-1, 1)
        ).view(-1)
        # Update the position in the end, as that's when linear velocity is
        # available (important for semi-implicit Euler integration).
        position = body.position + linear_velocity * dtime_

        return position, orientation, linear_momentum, angular_momentum

    def detect_collision_in_time_window(self, starttime, endtime, body):
        # Compute a "lookahead" state (position -> pos, orientation -> rot,
        # linear momentum -> p, angular momentul -> L).
        pos, rot, p, L = self.compute_lookahead_state_update(starttime, endtime, body)
        # Detect collisions
        contact_inds, _, _ = detect_ground_plane_contacts(
            self.get_world_vertices(body.vertices, rot, pos)
        )
        return contact_inds is not None

    def integrate(self, simulator, dtime):

        # WARNING: This will fail if the simulator contains multiple bodies.
        # NOTE: The ground plane (or other colliding planes) are not treated as
        # additional bodies.

        # Determine if a collision would occur if we rollout the current timestep.
        is_colliding_in_this_timestep = self.detect_collision_in_time_window(
            simulator.time, simulator.time + dtime, simulator.bodies[0]
        )

        # Compute forces (and torques).
        derivatives = simulator.compute_state_derivatives()
        # Compute state updates.
        for body, dstate in zip(simulator.bodies, derivatives):
            # body.position = body.position + dstate[0] * dtime
            body.orientation = body.orientation + dstate[1] * dtime
            body.linear_momentum = body.linear_momentum + dstate[2] * dtime
            body.angular_momentum = body.angular_momentum + dstate[3] * dtime
            body.orientation = normalize(body.orientation)
            body.linear_velocity = body.linear_momentum / body.masses.sum()
            inertia_world_inv = body.compute_inertia_world(
                body.inertia_body_inv, quaternion_to_rotmat(body.orientation)
            )
            body.angular_velocity = torch.matmul(
                inertia_world_inv, body.angular_momentum.view(-1, 1)
            ).view(-1)
            # Update the position in the end, as that's when linear velocity is
            # available.
            body.position = body.position + body.linear_velocity * dtime

        # Handle contact events (apply a corrective impluse)!
        if is_colliding_in_this_timestep:
            (
                contact_inds,
                contact_points,
                contact_normals,
            ) = detect_ground_plane_contacts(body.get_world_vertices())
            if contact_inds is None:
                raise ValueError(
                    "An error has occured! Execution reached here because a contact "
                    "event was discovered in the first place, and a toi fix has been "
                    "applied, and now an impulse response needs to be computed."
                )
            else:
                # Since we are colliding with the ground plane, the relative velocity
                # is given by dot_product(groundplane_normal, vel_object - vel_ground).
                # We assume the ground plane (outwards) normal to be (0, 1, 0) and the
                # velocity of the ground plane to be (0, 0, 0). Further, we assume that
                # the mass of the ground plane is very large (so that its inverse mass
                # is zero). We follow equation (8-18) on page 17 of Witkin and Baraff's
                # SIGGRAPH 1997 course notes on "Physically based modeling: principles
                # and practice" (Rigid body dynamics - Lecture Notes II (motion with
                # constraints)) (http://www.cs.cmu.edu/~baraff/sigcourse/notesd2.pdf).

                # Get the velocity of contact vertices (linear velocity +
                # cross_product(angular velocity, contact_vertex_in_world_frame -
                # center_of_mass_in_world_frame)). (Eq. (8-1) Page 11 of above notes).

                for idx, val in enumerate(contact_inds):
                    r = contact_points[idx] - body.position
                    n = contact_normals[idx]
                    vrel = body.linear_velocity + torch.cross(body.angular_velocity, r)
                    vrel = torch.dot(n, vrel)

                    if vrel > simulator.relative_velocity_threshold:
                        continue
                    if vrel > -simulator.relative_velocity_threshold:
                        continue

                    if simulator.verbose:
                        print("Apply corrective impulse here!")

                    num = -(1 + body.restitution) * vrel
                    minv = 1 / (body.masses.sum())
                    iinv = body.compute_inertia_world(
                        body.inertia_body_inv, quaternion_to_rotmat(body.orientation)
                    )
                    term0 = torch.cross(r, n)
                    term1 = torch.matmul(iinv, term0.unsqueeze(-1)).squeeze(-1)
                    term2 = torch.cross(term1, r)
                    term3 = torch.dot(n, term2)
                    den = minv + term3
                    j = (num / den) * n

                    body.linear_momentum = body.linear_momentum + j
                    body.angular_momentum = body.angular_momentum + torch.cross(r, j)
                    body.linear_velocity = body.linear_momentum / body.masses.sum()
                    body.angular_velocity = torch.matmul(
                        inertia_world_inv, body.angular_momentum.view(-1, 1)
                    ).view(-1)

        return dtime

# This code contains NVIDIA Confidential Information and is disclosed to you
# under a form of NVIDIA software license agreement provided separately to you.

# Notice
# NVIDIA Corporation and its licensors retain all intellectual property and
# proprietary rights in and to this software and related documentation and
# any modifications thereto. Any use, reproduction, disclosure, or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.

# ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
# NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
# THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
# MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

# Information and code furnished is believed to be accurate and reliable.
# However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
# information or for any infringement of patents or other rights of third parties that may
# result from its use. No license is granted by implication or otherwise under any patent
# or patent rights of NVIDIA Corporation. Details are subject to change without notice.
# This code supersedes and replaces all information previously supplied.
# NVIDIA Corporation products are not authorized for use as critical
# components in life support devices or systems without express written approval of
# NVIDIA Corporation.

# Copyright (c) 2020-2021 NVIDIA Corporation. All rights reserved.

import math
import torch
import numpy as np

from .util import *

GEO_SPHERE = 0
GEO_BOX = 1
GEO_CAPSULE = 2
GEO_MESH = 3
GEO_SDF = 4
GEO_PLANE = 5
GEO_NONE = 6       # disabled shape


class Mesh:
    def __init__(self, vertices, indices):

        self.vertices = vertices
        self.indices = indices

        # compute com and inertia (using density=1.0)
        com = np.mean(vertices, 0)

        num_tris = int(len(indices) / 3)

        # compute signed inertia for each tetrahedron
        # formed with the interior point, using an order-2
        # quadrature: https://www.sciencedirect.com/science/article/pii/S0377042712001604#br000040

        weight = 0.25
        alpha = math.sqrt(5.0) / 5.0

        I = np.zeros((3, 3))
        mass = 0.0

        for i in range(num_tris):

            p = np.array(vertices[indices[i * 3 + 0]])
            q = np.array(vertices[indices[i * 3 + 1]])
            r = np.array(vertices[indices[i * 3 + 2]])

            mid = (com + p + q + r) / 4.0

            pcom = p - com
            qcom = q - com
            rcom = r - com

            Dm = np.matrix((pcom, qcom, rcom)).T
            volume = np.linalg.det(Dm) / 6.0

            # quadrature points lie on the line between the
            # centroid and each vertex of the tetrahedron
            quads = (mid + (p - mid) * alpha, mid + (q - mid) * alpha, mid + (r - mid) * alpha, mid + (com - mid) * alpha)

            for j in range(4):

                # displacement of quadrature point from COM
                d = quads[j] - com

                I += weight * volume * (length_sq(d) * np.eye(3, 3) - np.outer(d, d))
                mass += weight * volume

        self.I = I
        self.mass = mass
        self.com = com


JOINT_PRISMATIC = 0
JOINT_REVOLUTE = 1
JOINT_FIXED = 2
JOINT_FREE = 3

# Notation
# =========
#
# We follow the conventions of Modern Robotics (Park & Lynch)
# all coordinate frames are right-handed. Subscripts refer to
# the space the vector is expressed in.
#
# X_sb refers to the SE(3) frame 'b' expressed in 's' coordinates
#
# Transforming a frame:
#
# X_sa*X_ab = X_sb
#
# Transforming a vector:
#
# X_sa*v_a = v_s


class Articulation:
    def __init__(self):
        pass


class ArticulationBuilder:
    def __init__(self):

        self.link_count = 0

        self.joint_parent = []         # index of the parent body                      (constant)
        self.joint_child = []          # index of the child body                       (constant)
        self.joint_axis = []           # joint axis in child joint frame               (constant)
        self.joint_X_pj = []           # frame of joint in parent                      (constant)
        self.joint_X_cm = []           # frame of child com (in child coordinates)     (constant)
        self.joint_S_s = []            # joint motion subspace in inertial frame       (fk)

        self.joint_q_start = []        # joint offset in the q array
        self.joint_qd_start = []       # joint offset in the qd array
        self.joint_qd_count = []       # number of dofs for this joint
        self.joint_type = []

        self.joint_q = []    # generalized coordinates       (input)
        self.joint_qd = []   # generalized velocities        (input)
        self.joint_qdd = []  # generalized accelerations     (id,fd)
        self.joint_tau = []  # generalized actuation         (input)
        self.joint_u = []    # generalized total torque      (fd)

        self.body_X_sc = []  # body frame                    (fk)
        self.body_X_sm = []  # mass frame                    (fk)

        # spatial quantities all expressed in global (spatial,inertial) coordinates

        self.body_v_s = []             # body velocity                 (id)
        self.body_a_s = []             # body acceleration             (id)
        self.body_f_s = []             # body force                    (id)
        self.body_f_ext_s = []         # body external force           (id)

        self.body_I_m = []   # body inertia (mass frame)     (constant)
        self.body_I_s = []   # body inertia (space frame)    (fk)

    def add_link(self, parent, X_pj, X_cm, axis, type, X_sm, inertia_m, mass):

        # body data
        self.body_X_sc.append(X_sm)
        self.body_X_sm.append(X_sm)
        self.body_I_m.append(spatial_matrix_from_inertia(inertia_m, mass))
        self.body_I_s.append(spatial_matrix())
        self.body_v_s.append(spatial_vector())
        self.body_a_s.append(spatial_vector())
        self.body_f_s.append(spatial_vector())

        # joint data
        self.joint_parent.append(parent)
        self.joint_X_pj.append(X_pj)
        self.joint_X_cm.append(X_cm)

        self.joint_q_start.append(len(self.joint_q))

        dofs = 0

        if (type == JOINT_PRISMATIC):
            self.joint_q.append(0.0)
            dofs = 1
        elif (type == JOINT_REVOLUTE):
            self.joint_q.append(0.0)
            dofs = 1
        elif (type == JOINT_FIXED):
            dofs = 0
        elif (type == JOINT_FREE):

            # translation
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)

            # quaternion
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(1.0)
            dofs = 6

        self.joint_type.append(type)
        self.joint_axis.append(np.array(axis))
        self.joint_qd_start.append(len(self.joint_qd))
        self.joint_qd_count.append(dofs)

        for i in range(dofs):
            self.joint_qd.append(0.0)
            self.joint_qdd.append(0.0)
            self.joint_tau.append(0.0)
            self.joint_u.append(0.0)
            self.joint_S_s.append(spatial_vector())

        self.link_count += 1
        return self.link_count - 1

    def finalize(self, adapter):

        a = Articulation()

        # one joint per-link
        a.link_count = self.link_count

        a.joint_type = torch.tensor(self.joint_type, dtype=torch.int32, device=adapter)
        a.joint_parent = torch.tensor(self.joint_parent, dtype=torch.int32, device=adapter)
        a.joint_X_pj = torch.tensor(transform_flatten_list(self.joint_X_pj), dtype=torch.float32, device=adapter)
        a.joint_X_cm = torch.tensor(transform_flatten_list(self.joint_X_cm), dtype=torch.float32, device=adapter)
        a.joint_axis = torch.tensor(self.joint_axis, dtype=torch.float32, device=adapter)
        a.joint_S_s = torch.tensor(self.joint_S_s, dtype=torch.float32, device=adapter)

        a.joint_q_start = torch.tensor(self.joint_q_start, dtype=torch.int32, device=adapter)
        a.joint_qd_start = torch.tensor(self.joint_qd_start, dtype=torch.int32, device=adapter)

        a.joint_q = torch.tensor(self.joint_q, dtype=torch.float32, device=adapter, requires_grad=True)
        a.joint_qd = torch.tensor(self.joint_qd, dtype=torch.float32, device=adapter, requires_grad=True)
        a.joint_qdd = torch.tensor(self.joint_qdd, dtype=torch.float32, device=adapter, requires_grad=True)
        a.joint_tau = torch.tensor(self.joint_tau, dtype=torch.float32, device=adapter, requires_grad=True)
        a.joint_u = torch.tensor(self.joint_u, dtype=torch.float32, device=adapter, requires_grad=True)

        a.body_X_sc = torch.tensor(transform_flatten_list(self.body_X_sc), dtype=torch.float32, device=adapter)
        a.body_X_sm = torch.tensor(transform_flatten_list(self.body_X_sm), dtype=torch.float32, device=adapter)

        a.body_v_s = torch.tensor(self.body_v_s, dtype=torch.float32, device=adapter)
        a.body_a_s = torch.tensor(self.body_a_s, dtype=torch.float32, device=adapter)
        a.body_f_s = torch.tensor(self.body_f_s, dtype=torch.float32, device=adapter)
        a.body_f_ext_s = torch.tensor(self.body_f_ext_s, dtype=torch.float32, device=adapter)

        a.body_I_m = torch.tensor(self.body_I_m, dtype=torch.float32, device=adapter)
        a.body_I_s = torch.tensor(self.body_I_s, dtype=torch.float32, device=adapter)

        # placeholder offset tensors for the articulation (could be in a larger flattened array of articulations)
        a.articulation_start = torch.tensor((0), dtype=torch.int32, device=adapter)
        a.articulation_end = torch.tensor((self.link_count), dtype=torch.int32, device=adapter)

        return a

    # compute transform across a joint
    def jcalc_transform(self, type, axis, joint_q, start):

        if (type == JOINT_REVOLUTE):

            q = joint_q[start]
            X_jc = transform((0.0, 0.0, 0.0), quat_from_axis_angle(axis, q))

        if (type == JOINT_PRISMATIC):

            q = joint_q[start]
            X_jc = transform(axis * q, quat_identity())

        if (type == JOINT_FREE):

            px = joint_q[start + 0]
            py = joint_q[start + 1]
            pz = joint_q[start + 2]

            qx = joint_q[start + 3]
            qy = joint_q[start + 4]
            qz = joint_q[start + 5]
            qw = joint_q[start + 6]

            X_jc = transform((px, py, pz), (qx, qy, qz, qw))

        return X_jc

    # compute motion subspace for a joint
    def jcalc_motion(self, type, axis, X_sc, S_s, start):

        if (type == JOINT_REVOLUTE):
            S_s[start] = transform_twist(X_sc, spatial_vector((*axis, 0.0, 0.0, 0.0)))

        if (type == JOINT_PRISMATIC):
            S_s[start] = transform_twist(X_sc, spatial_vector((0.0, 0.0, 0.0, *axis)))

        if (type == JOINT_FREE):
            S_s[start + 0] = transform_twist(X_sc, spatial_vector((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
            S_s[start + 1] = transform_twist(X_sc, spatial_vector((0.0, 1.0, 0.0, 0.0, 0.0, 0.0)))
            S_s[start + 2] = transform_twist(X_sc, spatial_vector((0.0, 0.0, 1.0, 0.0, 0.0, 0.0)))
            S_s[start + 3] = transform_twist(X_sc, spatial_vector((0.0, 0.0, 0.0, 1.0, 0.0, 0.0)))
            S_s[start + 4] = transform_twist(X_sc, spatial_vector((0.0, 0.0, 0.0, 0.0, 1.0, 0.0)))
            S_s[start + 5] = transform_twist(X_sc, spatial_vector((0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))

    # compute the velocity across a joint
    def jcalc_velocity(self, type, S_s, joint_qd, start):

        if (type == JOINT_REVOLUTE):
            v_j_s = S_s[start] * joint_qd[start]

        if (type == JOINT_PRISMATIC):
            v_j_s = S_s[start] * joint_qd[start]

        if (type == JOINT_FREE):
            v_j_s = S_s[start + 0] * joint_qd[start + 0]
            v_j_s += S_s[start + 1] * joint_qd[start + 1]
            v_j_s += S_s[start + 2] * joint_qd[start + 2]
            v_j_s += S_s[start + 3] * joint_qd[start + 3]
            v_j_s += S_s[start + 4] * joint_qd[start + 4]
            v_j_s += S_s[start + 5] * joint_qd[start + 5]

        return v_j_s

    # computes joint space forces/torques in tau
    def jcalc_tau(self, type, S_s, start, body_f_s, tau):

        if (type == JOINT_REVOLUTE):
            tau[start] = spatial_dot(S_s[start], body_f_s)

        if (type == JOINT_PRISMATIC):
            tau[start] = spatial_dot(S_s[start], body_f_s)

        if (type == JOINT_FREE):
            tau[start + 0] = spatial_dot(S_s[start + 0], body_f_s)
            tau[start + 1] = spatial_dot(S_s[start + 1], body_f_s)
            tau[start + 2] = spatial_dot(S_s[start + 2], body_f_s)
            tau[start + 3] = spatial_dot(S_s[start + 3], body_f_s)
            tau[start + 4] = spatial_dot(S_s[start + 4], body_f_s)
            tau[start + 5] = spatial_dot(S_s[start + 5], body_f_s)

    def jcalc_integrate(self, type, q, qd, qdd, q_start, qd_start, dt):

        if (type == JOINT_REVOLUTE):
            qd[qd_start] += qdd[qd_start] * dt
            q[q_start] += qd[qd_start] * dt

        if (type == JOINT_PRISMATIC):
            qd[qd_start] += qdd[qd_start] * dt
            q[q_start] += qd[qd_start] * dt

        if (type == JOINT_FREE):

            # integrate accelerations
            qd[qd_start + 0] += qdd[qd_start + 0] * dt
            qd[qd_start + 1] += qdd[qd_start + 1] * dt
            qd[qd_start + 2] += qdd[qd_start + 2] * dt
            qd[qd_start + 3] += qdd[qd_start + 3] * dt
            qd[qd_start + 4] += qdd[qd_start + 4] * dt
            qd[qd_start + 5] += qdd[qd_start + 5] * dt

            # linear vel (note q/qd switch order of linear angular elements)
            q[q_start + 0] += qd[qd_start + 3] * dt
            q[q_start + 1] += qd[qd_start + 4] * dt
            q[q_start + 2] += qd[qd_start + 5] * dt

            # angular vel
            w = np.array([qd[qd_start + 0], qd[qd_start + 1], qd[qd_start + 2]])

            # quat and quat derivative
            r = quat(q[q_start + 3], q[q_start + 4], q[q_start + 5], q[q_start + 6])
            drdt = quat_multiply((*w, 0.0), r) * 0.5

            # new orientation (normalized)
            r_new = normalize(r + drdt * dt)

            q[q_start + 3] = r_new[0]
            q[q_start + 4] = r_new[1]
            q[q_start + 5] = r_new[2]
            q[q_start + 6] = r_new[3]

    # forward kinematics, given q computes spatial transforms, inertias, and Jacobian
    def fk(self):

        with ScopedTimer("foward kinematics", False):

            # compute body transforms (excluding base which is prescribed)
            for i in range(0, self.link_count):

                # parent transform
                parent = self.joint_parent[i]

                # parent transform in spatial coordinates
                X_sp = transform_identity()
                if (parent > -1):
                    X_sp = self.body_X_sc[parent]

                type = self.joint_type[i]
                axis = self.joint_axis[i]
                coord_start = self.joint_q_start[i]
                dof_start = self.joint_qd_start[i]

                # compute transform across joint
                X_jc = self.jcalc_transform(type, axis, self.joint_q, coord_start)

                # compute child world transform
                X_pj = self.joint_X_pj[i]
                X_sc = transform_multiply(X_sp, transform_multiply(X_pj, X_jc))

                # compute transform of center of mass
                X_cm = self.joint_X_cm[i]
                X_sm = transform_multiply(X_sc, X_cm)

                # store geometry transforms
                self.body_X_sc[i] = X_sc
                self.body_X_sm[i] = X_sm

    # inverse dynamics, given q, qd, qdd computes tau, assumes that fk() has already been called
    def id(self, gravity=(0.0, -9.81, 0.0)):

        with ScopedTimer("inverse dynamics", False):

            tau = np.zeros(len(self.joint_qd))

            # compute body velocities, accelerations and forces
            for i in range(self.link_count):

                type = self.joint_type[i]
                axis = self.joint_axis[i]
                coord_start = self.joint_q_start[i]
                dof_start = self.joint_qd_start[i]

                X_sc = self.body_X_sc[i]

                # compute motion subspace in space frame (stores into the joint_S_s array)
                self.jcalc_motion(type, axis, X_sc, self.joint_S_s, dof_start)

                # velocity across the joint
                v_j_s = self.jcalc_velocity(type, self.joint_S_s, self.joint_qd, dof_start)

                # parent velocity
                parent = self.joint_parent[i]

                v_parent_s = spatial_vector()
                a_parent_s = spatial_vector()

                if (parent != -1):
                    v_parent_s = self.body_v_s[parent]
                    a_parent_s = self.body_a_s[parent]

                # body velocity, acceleration
                v_s = v_parent_s + v_j_s
                a_s = a_parent_s + spatial_cross(v_s, v_j_s)         # + self.joint_S_s[i]*self.joint_qdd[i]

                # compute body forces
                X_sm = self.body_X_sm[i]
                I_m = self.body_I_m[i]

                # gravity and external forces (expressed in frame aligned with s but centered at body mass)
                m = I_m[3, 3]
                f_ext_m = spatial_vector((0.0, 0.0, 0.0, *gravity)) * m
                f_ext_s = translate_wrench(X_sm[0], f_ext_m)

                # body forces
                I_s = transform_inertia(X_sm, I_m)

                f_b_s = np.dot(I_s, a_s) + spatial_cross_dual(v_s, np.dot(I_s, v_s))

                self.body_v_s[i] = v_s
                self.body_a_s[i] = a_s
                self.body_f_s[i] = f_b_s - f_ext_s
                self.body_I_s[i] = I_s

            # compute tau (backwards)
            for i in reversed(range(self.link_count)):

                type = self.joint_type[i]
                parent = self.joint_parent[i]

                self.jcalc_tau(type, self.joint_S_s, self.joint_qd_start[i], self.body_f_s[i], tau)

                # update parent
                self.body_f_s[parent] += self.body_f_s[i]

            return tau

    # space Jacobian
    def jacobian(self):

        with ScopedTimer("jacobian", False):

            dof_count = len(self.joint_qd)
            link_count = self.link_count

            J = np.zeros((link_count * 6, dof_count))

            for i in range(self.link_count):

                row_start = i * 6

                j = i
                while (j != -1):

                    dof_start = self.joint_qd_start[j]
                    dof_count = self.joint_qd_count[j]

                    for col in range(dof_start, dof_start + dof_count):
                        J[row_start:row_start + 6, col] = self.joint_S_s[col]

                    # walk up tree
                    j = self.joint_parent[j]

            return J

    # mass matrix
    def mass(self):

        with ScopedTimer("jacobian", False):

            n = self.link_count
            M = np.zeros((n * 6, n * 6))

            for i in range(n):

                # diagonal sub-block
                row_start = i * 6
                row_end = row_start + 6

                M[row_start:row_end, row_start:row_end] = self.body_I_s[i]

            return M

    def crba(self):
        J = self.jacobian()
        M = self.mass()

        with ScopedTimer("multiply", False):

            H = np.dot(np.dot(np.transpose(J), M), J)
            return H

    def fd_crba(self, dt):

        c = self.id()
        M = self.crba()

        with ScopedTimer("solve", False):
            self.joint_qdd = np.linalg.solve(M, -c)        # negative bias forces pg. 103 Featherstone

        # symplectic Euler update joint coordinates
        for i in range(self.link_count):

            type = self.joint_type[i]
            q_start = self.joint_q_start[i]
            qd_start = self.joint_qd_start[i]

            self.jcalc_integrate(type, self.joint_q, self.joint_qd, self.joint_qdd, q_start, qd_start, dt)

    def fd_aba(self, dt, gravity=(0.0, -9.81, 0.0)):

        # temporary
        Dinv = np.zeros(self.joint_count)

        # reset accelerations and forces
        for i in range(self.body_count):
            self.body_a_s[i] *= 0.0
            self.body_f_s[i] *= 0.0

        # compute body velocities and accelerations (excluding the base)
        for i in range(self.joint_count):

            # velocity across the joint
            v_j_s = self.joint_S_s[i] * self.joint_qd[i]

            # parent velocity
            parent = self.joint_parent[i]
            child = self.joint_child[i]

            v_parent_s = self.body_v_s[parent]

            # body velocity, acceleration
            v_s = v_parent_s + v_j_s
            a_s = spatial_cross(v_s, v_j_s)

            self.body_v_s[child] = v_s
            self.body_a_s[child] = a_s

        # compute body forces and inertias
        for i in range(self.body_count):

            X_sm = self.body_X_sm[i]

            # gravity and external forces (expressed in frame aligned with s but centered at body mass)
            m = self.body_I_m[i][3, 3]
            f_ext_m = spatial_vector((0.0, 0.0, 0.0, *gravity)) * m
            f_ext_s = translate_wrench(X_sm[0], f_ext_m)

            # body forces
            v_s = self.body_v_s[i]
            a_s = self.body_a_s[i]
            I_s = transform_inertia(X_sm, self.body_I_m[i])

            f_b_s = spatial_cross_dual(v_s, np.dot(I_s, v_s))

            self.body_f_s[i] = f_b_s - f_ext_s
            self.body_I_s[i] = I_s

        # compute articulated inertias and forces
        for i in reversed(range(self.joint_count)):

            parent = self.joint_parent[i]
            child = self.joint_child[i]
            S_s = self.joint_S_s[i]

            I_s = self.body_I_s[child]
            a_s = self.body_a_s[child]

            U_s = np.dot(I_s, S_s)
            D = np.dot(S_s, U_s)

            if (math.fabs(D) > 1.e-7):
                Dinv[i] = 1.0 / D
            else:
                Dinv[i] = 0.0

            f_s = self.body_f_s[child]
            u = self.joint_tau[i] - np.dot(S_s, f_s)

            Ia = I_s - np.outer(U_s, Dinv[i] * U_s)
            pa = f_s + np.dot(Ia, a_s) + np.dot(U_s, Dinv[i] * u)

            self.body_I_s[parent] += Ia
            self.body_f_s[parent] += pa

            self.joint_u[i] = u

        # update floating bodies
        for i in range(self.body_count):

            if (self.body_floating[i] == True):

                # solve for base acceleration (6x6 inv)
                a_s = -spatial_solve(self.body_I_s[i], self.body_f_s[i])

                self.body_a_s[i] = a_s
                self.body_v_s[i] += a_s * dt

                # compute velocity at body mass frame
                X_sm = self.body_X_sm[i]
                v_m = translate_twist(-self.body_X_sm[i][0], self.body_v_s[i])

                new_pos = X_sm[0] + v_m[3:6] * dt
                new_rot = X_sm[1] + quat_multiply((*v_m[0:3], 0.0), X_sm[1]) * 0.5 * dt

                self.body_X_sm[i] = transform(new_pos, normalize(new_rot))
                self.body_X_sc[i] = self.body_X_sm[0]

        for i in range(self.joint_count):

            parent = self.joint_parent[i]
            child = self.joint_child[i]

            a = self.body_a_s[parent] + self.body_a_s[child]

            # recompute U_s
            I_s = self.body_I_s[child]
            S_s = self.joint_S_s[i]
            U_s = np.dot(I_s, S_s)

            qdd = Dinv[i] * (self.joint_u[i] - np.dot(U_s, a))

            self.body_a_s[child] = a + self.joint_S_s[i] * qdd

            self.joint_qdd[i] = qdd
            self.joint_qd[i] += self.joint_qdd[i] * dt
            self.joint_q[i] += self.joint_qd[i] * dt


class Model:
    def __init__(self, adapter):

        self.particle_radius = 0.1
        self.particle_count = 0
        self.particle_offset = 0

        self.rigid_count = 0
        self.rigid_offset = 0

        self.spring_count = 0

        self.gravity = torch.tensor((0.0, -9.8, 0.0), dtype=torch.float32, device=adapter)

        self.contact_distance = 0.1
        self.contact_ke = 1.e+3
        self.contact_kd = 0.0
        self.contact_kf = 1.e+3
        self.contact_mu = 0.5

        self.tri_ke = 100.0
        self.tri_ka = 100.0
        self.tri_kd = 10.0
        self.tri_kb = 100.0
        self.tri_drag = 0.0
        self.tri_lift = 0.0

        self.edge_ke = 100.0
        self.edge_kd = 0.0

        self.taichi_backend = None

        self.adapter = adapter

    def state(self):

        s = State()
        s.q = torch.clone(self.particle_x)
        s.u = torch.clone(self.particle_v)

        s.rigid_x = torch.clone(self.rigid_x)
        s.rigid_r = torch.clone(self.rigid_r)

        s.rigid_v = torch.clone(self.rigid_v)
        s.rigid_w = torch.clone(self.rigid_w)

        return s

    def flatten(self):

        tensors = []

        # build a list of all tensor attributes
        for attr, value in self.__dict__.items():
            if (torch.is_tensor(value)):
                tensors.append(value)

        return tensors

    # builds contacts
    def collide(self, state):

        body0 = []
        body1 = []
        point = []
        dist = []
        mat = []

        def add_contact(b0, b1, p0, d, m):
            body0.append(b0)
            body1.append(b1)
            point.append(p0)
            dist.append(d)
            mat.append(m)

        for i in range(self.shape_count):

            geo_type = self.shape_geo_type[i].item()

            if (geo_type == GEO_SPHERE):

                radius = self.shape_geo_scale[i][0].item()

                add_contact(self.shape_body[i], -1, (0.0, 0.0, 0.0), radius, i)

            elif (geo_type == GEO_CAPSULE):

                radius = self.shape_geo_scale[i][0].item()
                half_width = self.shape_geo_scale[i][1].item()

                add_contact(self.shape_body[i], -1, (-half_width, 0.0, 0.0), radius, i)
                add_contact(self.shape_body[i], -1, (half_width, 0.0, 0.0), radius, i)

            elif (geo_type == GEO_MESH):

                mesh = self.shape_geo_src[i]
                scale = self.shape_geo_scale[i]

                for v in mesh.vertices:

                    p = (v[0] * scale[0], v[1] * scale[1], v[2] * scale[2])

                    add_contact(self.shape_body[i], -1, p, 0.0, i)

        # send to torch
        self.contact_body0 = torch.tensor(body0, device=self.adapter)
        self.contact_body1 = torch.tensor(body1, device=self.adapter)
        self.contact_point0 = torch.tensor(point, device=self.adapter)
        self.contact_dist = torch.tensor(dist, device=self.adapter)
        self.contact_material = torch.tensor(mat, device=self.adapter)

        self.contact_count = len(body0)


class State:
    def __init__(self):
        pass

    def clone(self):

        copy = State()
        copy.q = torch.empty_like(self.q)
        copy.u = torch.empty_like(self.u)

        copy.rigid_x = torch.empty_like(self.rigid_x)
        copy.rigid_r = torch.empty_like(self.rigid_r)

        copy.rigid_v = torch.empty_like(self.rigid_v)
        copy.rigid_w = torch.empty_like(self.rigid_w)

        return copy

    def flatten(self):

        return [self.q, self.u, self.rigid_x, self.rigid_r, self.rigid_v, self.rigid_w]


class ModelBuilder:
    def __init__(self):

        # particles
        self.particle_x = []
        self.particle_v = []
        self.particle_mass = []

        # rigids
        self.rigid_x = []
        self.rigid_r = []
        self.rigid_v = []
        self.rigid_w = []
        self.rigid_com = []
        self.rigid_inertia = []
        self.rigid_mass = []

        # shapes
        self.shape_x = []
        self.shape_r = []
        self.shape_body = []
        self.shape_geo_type = []
        self.shape_geo_scale = []
        self.shape_geo_src = []
        self.shape_materials = []

        # geometry
        self.geo_meshes = []
        self.geo_sdfs = []

        # springs
        self.spring_indices = []
        self.spring_rest_length = []
        self.spring_stiffness = []
        self.spring_damping = []
        self.spring_control = []

        # triangles
        self.tri_indices = []
        self.tri_poses = []
        self.tri_activations = []

        # edges (bending)
        self.edge_indices = []
        self.edge_rest_angle = []

        # tetrahedra
        self.tet_indices = []
        self.tet_poses = []
        self.tet_activations = []
        self.tet_materials = []

    # rigids, register a rigid body and return its index.
    def add_rigid_body(self, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)):

        self.rigid_x.append(pos)
        self.rigid_r.append(rot)
        self.rigid_v.append(vel)
        self.rigid_w.append(omega)

        self.rigid_com.append(np.zeros(3))
        self.rigid_inertia.append(np.zeros((3, 3)))
        self.rigid_mass.append(0.0)

        return len(self.rigid_x) - 1

    # shapes
    def add_shape_plane(self, plane, ke=1.e+5, kd=1000.0, kf=1000.0, mu=0.5):
        self.add_shape(-1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), GEO_PLANE, plane, None, 0.0, ke, kd, kf, mu)

    def add_shape_sphere(self, body, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), radius=1.0, density=1000.0, ke=1.e+5, kd=1000.0, kf=1000.0, mu=0.5):
        self.add_shape(body, pos, rot, GEO_SPHERE, (radius, 0.0, 0.0, 0.0), None, density, ke, kd, kf, mu)

    def add_shape_box(self,
                      body,
                      pos=(0.0, 0.0, 0.0),
                      rot=(0.0, 0.0, 0.0, 1.0),
                      hx=0.5,
                      hy=0.5,
                      hz=0.5,
                      density=1000.0,
                      ke=1.e+5,
                      kd=1000.0,
                      kf=1000.0,
                      mu=0.5):
        self.add_shape(body, pos, rot, GEO_BOX, (hx, hy, hz, 0.0), None, density, ke, kd, kf, mu)

    def add_shape_capsule(self,
                          body,
                          pos=(0.0, 0.0, 0.0),
                          rot=(0.0, 0.0, 0.0, 1.0),
                          radius=1.0,
                          half_width=0.5,
                          density=1000.0,
                          ke=1.e+5,
                          kd=1000.0,
                          kf=1000.0,
                          mu=0.5):
        self.add_shape(body, pos, rot, GEO_CAPSULE, (radius, half_width, 0.0, 0.0), None, density, ke, kd, kf, mu)

    def add_shape_mesh(self,
                       body,
                       pos=(0.0, 0.0, 0.0),
                       rot=(0.0, 0.0, 0.0, 1.0),
                       mesh=None,
                       scale=(1.0, 1.0, 1.0),
                       density=1000.0,
                       ke=1.e+5,
                       kd=1000.0,
                       kf=1000.0,
                       mu=0.5):
        self.add_shape(body, pos, rot, GEO_MESH, (scale[0], scale[1], scale[2], 0.0), mesh, density, ke, kd, kf, mu)

    def add_shape(self, body, pos, rot, type, scale, src, density, ke, kd, kf, mu):
        self.shape_body.append(body)
        self.shape_x.append(pos)
        self.shape_r.append(rot)
        self.shape_geo_type.append(type)
        self.shape_geo_scale.append((scale[0], scale[1], scale[2], 0.0))
        self.shape_geo_src.append(src)
        self.shape_materials.append((ke, kd, kf, mu))

        (m, I) = self.compute_shape_mass(type, scale, src, density)

        self.update_rigid_mass(body, m, I, np.array(pos), np.array(rot))

    # particles
    def add_particle(self, pos, vel, mass):
        self.particle_x.append(pos)
        self.particle_v.append(vel)
        self.particle_mass.append(mass)

        return len(self.particle_x) - 1

    def add_spring(self, i, j, ke, kd, control):
        self.spring_indices.append(i)
        self.spring_indices.append(j)
        self.spring_stiffness.append(ke)
        self.spring_damping.append(kd)
        self.spring_control.append(control)

        # compute rest length
        p = self.particle_x[i]
        q = self.particle_x[j]

        delta = np.subtract(p, q)
        l = np.sqrt(np.dot(delta, delta))

        self.spring_rest_length.append(l)

    def add_triangle(self, i, j, k):

        # compute basis for 2D rest pose
        p = np.array(self.particle_x[i])
        q = np.array(self.particle_x[j])
        r = np.array(self.particle_x[k])

        qp = q - p
        rp = r - p

        # construct basis aligned with the triangle
        n = normalize(np.cross(qp, rp))
        e1 = normalize(qp)
        e2 = normalize(np.cross(n, e1))

        R = np.matrix((e1, e2))
        M = np.matrix((qp, rp))

        D = R * M.T
        inv_D = np.linalg.inv(D)

        area = np.linalg.det(D) / 2.0

        if (area < 0.0):
            print("inverted triangle element")

        self.tri_indices.append((i, j, k))
        self.tri_poses.append(inv_D.tolist())
        self.tri_activations.append(0.0)

        return area

    def add_tetrahedron(self, i, j, k, l, k_mu=1.e+3, k_lambda=1.e+3, k_damp=0.0):

        # compute basis for 2D rest pose
        p = np.array(self.particle_x[i])
        q = np.array(self.particle_x[j])
        r = np.array(self.particle_x[k])
        s = np.array(self.particle_x[l])

        qp = q - p
        rp = r - p
        sp = s - p

        Dm = np.matrix((qp, rp, sp)).T
        volume = np.linalg.det(Dm) / 6.0

        if (volume <= 0.0):
            print("inverted tetrahedral element")
        else:

            inv_Dm = np.linalg.inv(Dm)

            self.tet_indices.append((i, j, k, l))
            self.tet_poses.append(inv_Dm.tolist())
            self.tet_activations.append(0.0)
            self.tet_materials.append((k_mu, k_lambda, k_damp))

        return volume

    def add_edge(self, i, j, k, l, rest=None):

        # compute rest angle
        if (rest == None):

            x1 = np.array(self.particle_x[i])
            x2 = np.array(self.particle_x[j])
            x3 = np.array(self.particle_x[k])
            x4 = np.array(self.particle_x[l])

            n1 = normalize(np.cross(x3 - x1, x4 - x1))
            n2 = normalize(np.cross(x4 - x2, x3 - x2))
            e = normalize(x4 - x3)

            d = np.clip(np.dot(n2, n1), -1.0, 1.0)

            angle = math.acos(d)
            sign = np.sign(np.dot(np.cross(n2, n1), e))

            rest = angle * sign

        self.edge_indices.append((i, j, k, l))
        self.edge_rest_angle.append(rest)

    def grid_index(self, x, y, dim_x):
        return y * dim_x + x

    def add_cloth_grid(self,
                       pos,
                       rot,
                       vel,
                       dim_x,
                       dim_y,
                       cell_x,
                       cell_y,
                       mass,
                       reverse_winding=False,
                       fix_left=False,
                       fix_right=False,
                       fix_top=False,
                       fix_bottom=False):

        start_vertex = len(self.particle_x)
        start_tri = len(self.tri_indices)

        for y in range(0, dim_y + 1):
            for x in range(0, dim_x + 1):

                g = np.array((x * cell_x, y * cell_y, 0.0))
                p = quat_rotate(rot, g) + pos
                m = mass

                if (x == 0 and fix_left):
                    m = 0.0
                elif (x == dim_x and fix_right):
                    m = 0.0
                elif (y == 0 and fix_bottom):
                    m = 0.0
                elif (y == dim_y and fix_top):
                    m = 0.0

                self.add_particle(p, vel, m)

                if (x > 0 and y > 0):

                    if (reverse_winding):
                        tri1 = (start_vertex + self.grid_index(x - 1, y - 1, dim_x + 1),
                                start_vertex + self.grid_index(x, y - 1, dim_x + 1),
                                start_vertex + self.grid_index(x, y, dim_x + 1))

                        tri2 = (start_vertex + self.grid_index(x - 1, y - 1, dim_x + 1),
                                start_vertex + self.grid_index(x, y, dim_x + 1),
                                start_vertex + self.grid_index(x - 1, y, dim_x + 1))

                        self.add_triangle(*tri1)
                        self.add_triangle(*tri2)

                    else:

                        tri1 = (start_vertex + self.grid_index(x - 1, y - 1, dim_x + 1),
                                start_vertex + self.grid_index(x, y - 1, dim_x + 1),
                                start_vertex + self.grid_index(x - 1, y, dim_x + 1))

                        tri2 = (start_vertex + self.grid_index(x, y - 1, dim_x + 1),
                                start_vertex + self.grid_index(x, y, dim_x + 1),
                                start_vertex + self.grid_index(x - 1, y, dim_x + 1))

                        self.add_triangle(*tri1)
                        self.add_triangle(*tri2)

        end_vertex = len(self.particle_x)
        end_tri = len(self.tri_indices)

        # bending constraints, could create these explicitly for a grid but this
        # is a good test of the adjacency structure
        adj = MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        for k, e in adj.edges.items():

            # skip open edges
            if (e.f0 == -1 or e.f1 == -1):
                continue

            self.add_edge(e.o0, e.o1, e.v0, e.v1)          # opposite 0, opposite 1, vertex 0, vertex 1

    def add_cloth_mesh(self, pos, rot, scale, vel, vertices, indices, density, edge_callback=None, face_callback=None):

        num_tris = int(len(indices) / 3)

        start_vertex = len(self.particle_x)
        start_tri = len(self.tri_indices)

        # particles
        for i, v in enumerate(vertices):

            p = quat_rotate(rot, v * scale) + pos

            self.add_particle(p, vel, 0.0)

        # triangles
        for t in range(num_tris):

            i = start_vertex + indices[t * 3 + 0]
            j = start_vertex + indices[t * 3 + 1]
            k = start_vertex + indices[t * 3 + 2]

            if (face_callback):
                face_callback(i, j, k)

            area = self.add_triangle(i, j, k)

            # add area fraction to particles
            if (area > 0.0):

                self.particle_mass[i] += density * area / 3.0
                self.particle_mass[j] += density * area / 3.0
                self.particle_mass[k] += density * area / 3.0

        end_vertex = len(self.particle_x)
        end_tri = len(self.tri_indices)

        adj = MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        # bend constraints
        for k, e in adj.edges.items():

            # skip open edges
            if (e.f0 == -1 or e.f1 == -1):
                continue

            if (edge_callback):
                edge_callback(e.f0, e.f1)

            self.add_edge(e.o0, e.o1, e.v0, e.v1)

    def add_soft_grid(self,
                      pos,
                      rot,
                      vel,
                      dim_x,
                      dim_y,
                      dim_z,
                      cell_x,
                      cell_y,
                      cell_z,
                      density,
                      k_mu,
                      k_lambda,
                      k_damp,
                      reverse_winding=False,
                      fix_left=False,
                      fix_right=False,
                      fix_top=False,
                      fix_bottom=False):

        start_vertex = len(self.particle_x)

        mass = cell_x * cell_y * cell_z * density / 8.0

        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):

                    v = np.array((x * cell_x, y * cell_y, z * cell_z))
                    m = mass

                    if (fix_left and x == 0):
                        m = 0.0

                    if (fix_right and x == dim_x):
                        m = 0.0

                    if (fix_top and y == dim_y):
                        m = 0.0

                    if (fix_bottom and y == 0):
                        m = 0.0

                    p = quat_rotate(rot, v) + pos

                    self.add_particle(p, vel, m)

        # dict of open faces
        faces = {}

        def add_face(i, j, k):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        def add_tet(i, j, k, l):
            self.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

            add_face(i, k, j)
            add_face(j, k, l)
            add_face(i, j, l)
            add_face(i, l, k)

        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):

                    v0 = grid_index(x, y, z) + start_vertex
                    v1 = grid_index(x + 1, y, z) + start_vertex
                    v2 = grid_index(x + 1, y, z + 1) + start_vertex
                    v3 = grid_index(x, y, z + 1) + start_vertex
                    v4 = grid_index(x, y + 1, z) + start_vertex
                    v5 = grid_index(x + 1, y + 1, z) + start_vertex
                    v6 = grid_index(x + 1, y + 1, z + 1) + start_vertex
                    v7 = grid_index(x, y + 1, z + 1) + start_vertex

                    if (((x & 1) ^ (y & 1) ^ (z & 1))):

                        add_tet(v0, v1, v4, v3)
                        add_tet(v2, v3, v6, v1)
                        add_tet(v5, v4, v1, v6)
                        add_tet(v7, v6, v3, v4)
                        add_tet(v4, v1, v6, v3)

                    else:

                        add_tet(v1, v2, v5, v0)
                        add_tet(v3, v0, v7, v2)
                        add_tet(v4, v7, v0, v5)
                        add_tet(v6, v5, v2, v7)
                        add_tet(v5, v2, v7, v0)

        # add triangles
        for k, v in faces.items():
            self.add_triangle(v[0], v[1], v[2])

    def add_soft_mesh(self, pos, rot, scale, vel, vertices, indices, density, k_mu, k_lambda, k_damp):

        num_tets = int(len(indices) / 4)

        start_vertex = len(self.particle_x)
        start_tri = len(self.tri_indices)

        # dict of open faces
        faces = {}

        def add_face(i, j, k):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        # add particles
        for v in vertices:

            p = quat_rotate(rot, v * scale) + pos

            self.add_particle(p, vel, 0.0)

        # add tetrahedra
        for t in range(num_tets):

            v0 = start_vertex + indices[t * 4 + 0]
            v1 = start_vertex + indices[t * 4 + 1]
            v2 = start_vertex + indices[t * 4 + 2]
            v3 = start_vertex + indices[t * 4 + 3]

            volume = self.add_tetrahedron(v0, v1, v2, v3, k_mu, k_lambda, k_damp)

            # distribute volume fraction to particles
            if (volume > 0.0):

                self.particle_mass[v0] += density * volume / 4.0
                self.particle_mass[v1] += density * volume / 4.0
                self.particle_mass[v2] += density * volume / 4.0
                self.particle_mass[v3] += density * volume / 4.0

                # build open faces
                add_face(v0, v2, v1)
                add_face(v1, v2, v3)
                add_face(v0, v1, v3)
                add_face(v0, v3, v2)

        # add triangles
        for k, v in faces.items():
            try:
                self.add_triangle(v[0], v[1], v[2])
            except np.linalg.LinAlgError:
                continue

    # geo
    def add_geo_mesh(self):
        pass

    def add_geo_sdf(self):
        pass

    # control
    def add_control(self):
        pass

    def compute_sphere_inertia(self, density, r):

        v = 4.0 / 3.0 * math.pi * r * r * r

        m = density * v
        Ia = 2.0 / 5.0 * m * r * r

        I = np.array([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])

        return (m, I)

    def compute_capsule_inertia(self, density, r, l):

        ms = density * (4.0 / 3.0) * math.pi * r * r * r
        mc = density * math.pi * r * r * l

        # total mass
        m = ms + mc

        # adapted from ODE
        Ia = mc * (0.25 * r * r + (1.0 / 12.0) * l * l) + ms * (0.4 * r * r + 0.375 * r * l + 0.25 * l * l)
        Ib = (mc * 0.5 + ms * 0.4) * r * r

        I = np.array([[Ib, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])

        return (m, I)

    def compute_box_inertia(self, density, w, h, d):

        v = w * h * d
        m = density * v

        Ia = 1.0 / 12.0 * m * (h * h + d * d)
        Ib = 1.0 / 12.0 * m * (w * w + d * d)
        Ic = 1.0 / 12.0 * m * (w * w + h * h)

        I = np.array([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ic]])

        return (m, I)

    def compute_shape_mass(self, type, scale, src, density):
        if density == 0:     # zero density means fixed
            return 0, np.zeros((3, 3))

        if (type == GEO_SPHERE):
            return self.compute_sphere_inertia(density, scale[0])
        elif (type == GEO_BOX):
            return self.compute_box_inertia(density, scale[0] * 2.0, scale[1] * 2.0, scale[2] * 2.0)
        elif (type == GEO_CAPSULE):
            return self.compute_capsule_inertia(density, scale[0], scale[1] * 2.0)
        elif (type == GEO_MESH):
            #todo: non-uniform scale of inertia tensor
            s = scale[0]     # eventually want to compute moment of inertia for mesh.
            return (density * src.mass * s * s * s, density * src.I * s * s * s * s * s)

    def transform_inertia(self, m, I, p, q):
        R = quat_to_matrix(q)

        # Steiner's theorem
        return R * I * R.T + m * (np.dot(p, p) * np.eye(3) - np.outer(p, p))

    # incrementally updates rigid body mass with additional mass and inertia expressed at a local to the body
    def update_rigid_mass(self, i, m, I, p, q):
        # find new COM
        new_mass = self.rigid_mass[i] + m

        if new_mass == 0:    # no mass
            return

        new_com = (self.rigid_com[i] * self.rigid_mass[i] + p * m) / new_mass

        # shift inertia to new COM
        com_offset = new_com - self.rigid_com[i]
        shape_offset = new_com - p

        new_inertia = self.transform_inertia(self.rigid_mass[i], self.rigid_inertia[i], com_offset, quat_identity()) + self.transform_inertia(
            m, I, shape_offset, q)

        self.rigid_mass[i] = new_mass
        self.rigid_inertia[i] = new_inertia
        self.rigid_com[i] = new_com

    # returns a (model, state) pair given the description
    def finalize(self, adapter):

        # construct particle inv masses
        particle_inv_mass = []
        for m in self.particle_mass:
            if (m > 0.0):
                particle_inv_mass.append(1.0 / m)
            else:
                particle_inv_mass.append(0.0)

        # construct rigid inv masses
        rigid_inv_mass = []
        rigid_inv_inertia = []

        for b in range(len(self.rigid_x)):
            if self.rigid_mass[b] > 0:
                rigid_inv_mass.append(1.0 / self.rigid_mass[b])
                rigid_inv_inertia.append(np.linalg.inv(self.rigid_inertia[b]))
            else:  # zero mass is fixed
                rigid_inv_mass.append(0.0)
                rigid_inv_inertia.append(np.zeros((3, 3)))

        #-------------------------------------
        # construct Model (non-time varying) data

        m = Model(adapter)

        m.particle_x = torch.tensor(self.particle_x, dtype=torch.float32, device=adapter)
        m.particle_v = torch.tensor(self.particle_v, dtype=torch.float32, device=adapter)
        m.particle_mass = torch.tensor(self.particle_mass, dtype=torch.float32, device=adapter)
        m.particle_inv_mass = torch.tensor(particle_inv_mass, dtype=torch.float32, device=adapter)

        m.rigid_com = torch.tensor(self.rigid_com, dtype=torch.float32, device=adapter)
        m.rigid_mass = torch.tensor(self.rigid_mass, dtype=torch.float32, device=adapter)
        m.rigid_inertia = torch.tensor(self.rigid_inertia, dtype=torch.float32, device=adapter)
        m.rigid_inv_mass = torch.tensor(rigid_inv_mass, dtype=torch.float32, device=adapter)
        m.rigid_inv_inertia = torch.tensor(rigid_inv_inertia, dtype=torch.float32, device=adapter)

        m.rigid_x = torch.tensor(self.rigid_x, dtype=torch.float32, device=adapter)
        m.rigid_r = torch.tensor(self.rigid_r, dtype=torch.float32, device=adapter)
        m.rigid_v = torch.tensor(self.rigid_v, dtype=torch.float32, device=adapter)
        m.rigid_w = torch.tensor(self.rigid_w, dtype=torch.float32, device=adapter)

        m.shape_position = torch.tensor(self.shape_x, dtype=torch.float32, device=adapter)
        m.shape_rotation = torch.tensor(self.shape_r, dtype=torch.float32, device=adapter)
        m.shape_body = torch.tensor(self.shape_body, dtype=torch.int32, device=adapter)
        m.shape_geo_type = torch.tensor(self.shape_geo_type, dtype=torch.int32, device=adapter)
        m.shape_geo_src = self.shape_geo_src     # torch.tensor(self.shape_geo_type, dtype=torch.int32, device=adapter)
        m.shape_geo_scale = torch.tensor(self.shape_geo_scale, dtype=torch.float32, device=adapter)
        m.shape_materials = torch.tensor(self.shape_materials, dtype=torch.float32, device=adapter)

        m.spring_indices = torch.tensor(self.spring_indices, dtype=torch.int32, device=adapter)
        m.spring_rest_length = torch.tensor(self.spring_rest_length, dtype=torch.float32, device=adapter)
        m.spring_stiffness = torch.tensor(self.spring_stiffness, dtype=torch.float32, device=adapter)
        m.spring_damping = torch.tensor(self.spring_damping, dtype=torch.float32, device=adapter)
        m.spring_control = torch.tensor(self.spring_control, dtype=torch.float32, device=adapter)

        m.tri_indices = torch.tensor(self.tri_indices, dtype=torch.int32, device=adapter)
        m.tri_poses = torch.tensor(self.tri_poses, dtype=torch.float32, device=adapter)
        m.tri_activations = torch.tensor(self.tri_activations, dtype=torch.float32, device=adapter)

        m.edge_indices = torch.tensor(self.edge_indices, dtype=torch.int32, device=adapter)
        m.edge_rest_angle = torch.tensor(self.edge_rest_angle, dtype=torch.float32, device=adapter)

        m.tet_indices = torch.tensor(self.tet_indices, dtype=torch.int32, device=adapter)
        m.tet_poses = torch.tensor(self.tet_poses, dtype=torch.float32, device=adapter)
        m.tet_activations = torch.tensor(self.tet_activations, dtype=torch.float32, device=adapter)
        m.tet_materials = torch.tensor(self.tet_materials, dtype=torch.float32, device=adapter)

        # set up offsets
        m.particle_count = len(self.particle_x)
        m.particle_offset = 0

        m.rigid_count = len(self.rigid_x)
        m.rigid_offset = m.particle_offset + m.particle_count * 3

        m.shape_count = len(self.shape_geo_type)
        m.tri_count = len(self.tri_poses)
        m.tet_count = len(self.tet_poses)
        m.edge_count = len(self.edge_rest_angle)
        m.spring_count = len(self.spring_rest_length)
        m.contact_count = 0
        m.tri_collisions = False

        # store refs to geometry
        m.geo_meshes = self.geo_meshes
        m.geo_sdfs = self.geo_sdfs

        # enable ground plane
        m.ground = True
        m.gravity = torch.tensor((0.0, -9.8, 0.0), dtype=torch.float32, device=adapter)

        # from dflex.config import use_taichi
        # if use_taichi:
        #     from dflex.taichi_backend import TaichiBackend
        #     m.taichi_backend = TaichiBackend(m)

        #-------------------------------------
        # construct generalized State (time-varying) vector

        return m

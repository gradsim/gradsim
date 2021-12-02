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
import time

import torch
import numpy as np

from . import util
from . import adjoint as df
from . import config
from .model import *

# Todo
#-----
#
# [x] Spring model
# [x] 2D FEM model
# [x] 3D FEM model
# [x] Cloth
#     [x] Wind/Drag model
#     [x] Bending model
#     [x] Triangle collision
# [x] Rigid body model
# [x] Rigid shape contact
#     [x] Sphere
#     [x] Capsule
#     [x] Box
#     [ ] Convex
#     [ ] SDF
# [ ] Implicit solver
# [x] USD import
# [x] USD export
# -----

# externally compiled kernels module (C++/CUDA code with PyBind entry points)
kernels = None

@df.func
def test(c: float):

    x = 1.0

    if (c < 3.0):
        x = 2.0

    return x*6.0



def kernel_init():
    global kernels
    if config.use_taichi:
        # Do not compile non-Taichi kernels when using Taichi
        class FakeKernels:
            def __init__(self):
                pass

            def __getattr__(self, item):
                def fake_func(*args, **kwargs):
                    pass

                return fake_func

        kernels = FakeKernels()
    else:
        kernels = df.compile()


@df.kernel
def integrate_particles(x: df.tensor(df.float3),
                        v: df.tensor(df.float3),
                        f: df.tensor(df.float3),
                        w: df.tensor(float),
                        gravity: df.tensor(df.float3),
                        dt: float,
                        x_new: df.tensor(df.float3),
                        v_new: df.tensor(df.float3)):

    tid = df.tid()

    x0 = df.load(x, tid)
    v0 = df.load(v, tid)
    f0 = df.load(f, tid)
    inv_mass = df.load(w, tid)

    g = df.load(gravity, 0)

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + g * df.step(0.0 - inv_mass)) * dt
    x1 = x0 + v1 * dt

    df.store(x_new, tid, x1)
    df.store(v_new, tid, v1)


# semi-implicit Euler integration
@df.kernel
def integrate_rigids(rigid_x: df.tensor(df.float3),
                     rigid_r: df.tensor(df.quat),
                     rigid_v: df.tensor(df.float3),
                     rigid_w: df.tensor(df.float3),
                     rigid_f: df.tensor(df.float3),
                     rigid_t: df.tensor(df.float3),
                     inv_m: df.tensor(float),
                     inv_I: df.tensor(df.mat33),
                     gravity: df.tensor(df.float3),
                     dt: float,
                     rigid_x_new: df.tensor(df.float3),
                     rigid_r_new: df.tensor(df.quat),
                     rigid_v_new: df.tensor(df.float3),
                     rigid_w_new: df.tensor(df.float3)):

    tid = df.tid()

    # positions
    x0 = df.load(rigid_x, tid)
    r0 = df.load(rigid_r, tid)

    # velocities
    v0 = df.load(rigid_v, tid)
    w0 = df.load(rigid_w, tid)         # angular velocity

    # forces
    f0 = df.load(rigid_f, tid)
    t0 = df.load(rigid_t, tid)

    # masses
    inv_mass = df.load(inv_m, tid)     # 1 / mass
    inv_inertia = df.load(inv_I, tid)  # inverse of 3x3 inertia matrix

    g = df.load(gravity, 0)

    # linear part
    v1 = v0 + (f0 * inv_mass + g * df.nonzero(inv_mass)) * dt           # linear integral (linear position/velocity)
    x1 = x0 + v1 * dt

    # angular part

    # so reverse multiplication by r0 takes you from global coordinates into local coordinates
    # because it's covector and thus gets pulled back rather than pushed forward
    wb = df.rotate_inv(r0, w0)         # angular integral (angular velocity and rotation), rotate into object reference frame
    tb = df.rotate_inv(r0, t0)         # also rotate torques into local coordinates

    # I^{-1} torque = angular acceleration and inv_inertia is always going to be in the object frame.
    # So we need to rotate into that frame, and then back into global.
    w1 = df.rotate(r0, wb + inv_inertia * tb * dt)                   # I^-1 * torque * dt., then go back into global coordinates
    r1 = df.normalize(r0 + df.quat(w1, 0.0) * r0 * 0.5 * dt)         # rotate around w1 by dt

    df.store(rigid_x_new, tid, x1)
    df.store(rigid_r_new, tid, r1)
    df.store(rigid_v_new, tid, v1)
    df.store(rigid_w_new, tid, w1)


@df.kernel
def eval_springs(x: df.tensor(df.float3),
                 v: df.tensor(df.float3),
                 spring_indices: df.tensor(int),
                 spring_rest_lengths: df.tensor(float),
                 spring_stiffness: df.tensor(float),
                 spring_damping: df.tensor(float),
                 f: df.tensor(df.float3)):

    tid = df.tid()

    i = df.load(spring_indices, tid * 2 + 0)
    j = df.load(spring_indices, tid * 2 + 1)

    ke = df.load(spring_stiffness, tid)
    kd = df.load(spring_damping, tid)
    rest = df.load(spring_rest_lengths, tid)

    xi = df.load(x, i)
    xj = df.load(x, j)

    vi = df.load(v, i)
    vj = df.load(v, j)

    xij = xi - xj
    vij = vi - vj

    l = length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = dot(dir, vij)

    # damping based on relative velocity.
    fs = dir * (ke * c + kd * dcdt)

    df.atomic_sub(f, i, fs)
    df.atomic_add(f, j, fs)


@df.kernel
def eval_triangles(x: df.tensor(df.float3),
                   v: df.tensor(df.float3),
                   indices: df.tensor(int),
                   pose: df.tensor(df.mat22),
                   activation: df.tensor(float),
                   k_mu: float,
                   k_lambda: float,
                   k_damp: float,
                   k_drag: float,
                   k_lift: float,
                   f: df.tensor(df.float3)):
    tid = df.tid()

    i = df.load(indices, tid * 3 + 0)
    j = df.load(indices, tid * 3 + 1)
    k = df.load(indices, tid * 3 + 2)

    p = df.load(x, i)        # point zero
    q = df.load(x, j)        # point one
    r = df.load(x, k)        # point two

    vp = df.load(v, i)       # vel zero
    vq = df.load(v, j)       # vel one
    vr = df.load(v, k)       # vel two

    qp = q - p     # barycentric coordinates (centered at p)
    rp = r - p

    Dm = df.load(pose, tid)

    inv_rest_area = df.determinant(Dm) * 2.0     # 1 / det(A) = det(A^-1)
    rest_area = 1.0 / inv_rest_area

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    k_damp = k_damp * rest_area

    # F = Xs*Xm^-1
    f1 = qp * Dm[0, 0] + rp * Dm[1, 0]
    f2 = qp * Dm[0, 1] + rp * Dm[1, 1]

    #-----------------------------
    # St. Venant-Kirchoff

    # # Green strain, F'*F-I
    # e00 = dot(f1, f1) - 1.0
    # e10 = dot(f2, f1)
    # e01 = dot(f1, f2)
    # e11 = dot(f2, f2) - 1.0

    # E = df.mat22(e00, e01,
    #              e10, e11)

    # # local forces (deviatoric part)
    # T = df.mul(E, df.transpose(Dm))

    # # spatial forces, F*T
    # fq = (f1*T[0,0] + f2*T[1,0])*k_mu*2.0
    # fr = (f1*T[0,1] + f2*T[1,1])*k_mu*2.0
    # alpha = 1.0

    #-----------------------------
    # Baraff & Witkin, note this model is not isotropic

    # c1 = length(f1) - 1.0
    # c2 = length(f2) - 1.0
    # f1 = normalize(f1)*c1*k1
    # f2 = normalize(f2)*c2*k1

    # fq = f1*Dm[0,0] + f2*Dm[0,1]
    # fr = f1*Dm[1,0] + f2*Dm[1,1]

    #-----------------------------
    # Neo-Hookean (with rest stability)

    # force = mu*F*Dm'
    fq = (f1 * Dm[0, 0] + f2 * Dm[0, 1]) * k_mu
    fr = (f1 * Dm[1, 0] + f2 * Dm[1, 1]) * k_mu
    alpha = 1.0 + k_mu / k_lambda

    #-----------------------------
    # Area Preservation

    n = df.cross(qp, rp)
    area = df.length(n) * 0.5

    # actuation
    act = df.load(activation, tid)

    # J-alpha
    c = area * inv_rest_area - alpha + act

    # dJdx
    n = df.normalize(n)
    dcdq = df.cross(rp, n) * inv_rest_area * 0.5
    dcdr = df.cross(n, qp) * inv_rest_area * 0.5

    f_area = k_lambda * c

    #-----------------------------
    # Area Damping

    dcdt = dot(dcdq, vq) + dot(dcdr, vr) - dot(dcdq + dcdr, vp)
    f_damp = k_damp * dcdt

    fq = fq + dcdq * (f_area + f_damp)
    fr = fr + dcdr * (f_area + f_damp)
    fp = fq + fr

    #-----------------------------
    # Lift + Drag

    vmid = (vp + vr + vq) * 0.3333
    vdir = df.normalize(vmid)

    f_drag = vmid * (k_drag * area * df.abs(df.dot(n, vmid)))
    f_lift = n * (k_lift * area * (1.57079 - df.acos(df.dot(n, vdir)))) * dot(vmid, vmid)

    # note reversed sign due to atomic_add below.. need to write the unary op -
    fp = fp - f_drag - f_lift
    fq = fq + f_drag + f_lift
    fr = fr + f_drag + f_lift

    # apply forces
    df.atomic_add(f, i, fp)
    df.atomic_sub(f, j, fq)
    df.atomic_sub(f, k, fr)

@df.func
def triangle_closest_point_barycentric(a: df.float3, b: df.float3, c: df.float3, p: df.float3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = df.dot(ab, ap)
    d2 = df.dot(ac, ap)

    if (d1 <= 0.0 and d2 <= 0.0):
        return float3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = df.dot(ab, bp)
    d4 = df.dot(ac, bp)

    if (d3 >= 0.0 and d4 <= d3):
        return float3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
        return float3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    if (d6 >= 0.0 and d5 <= d6):
        return float3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
        return float3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
        return float3(0.0, w, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return float3(1.0 - v - w, v, w)

# @df.func
# def triangle_closest_point(a: df.float3, b: df.float3, c: df.float3, p: df.float3):
#     ab = b - a
#     ac = c - a
#     ap = p - a

#     d1 = df.dot(ab, ap)
#     d2 = df.dot(ac, ap)

#     if (d1 <= 0.0 and d2 <= 0.0):
#         return a

#     bp = p - b
#     d3 = df.dot(ab, bp)
#     d4 = df.dot(ac, bp)

#     if (d3 >= 0.0 and d4 <= d3):
#         return b

#     vc = d1 * d4 - d3 * d2
#     v = d1 / (d1 - d3)
#     if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
#         return a + ab * v

#     cp = p - c
#     d5 = dot(ab, cp)
#     d6 = dot(ac, cp)

#     if (d6 >= 0.0 and d5 <= d6):
#         return c

#     vb = d5 * d2 - d1 * d6
#     w = d2 / (d2 - d6)
#     if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
#         return a + ac * w

#     va = d3 * d6 - d5 * d4
#     w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
#     if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
#         return b + (c - b) * w

#     denom = 1.0 / (va + vb + vc)
#     v = vb * denom
#     w = vc * denom

#     return a + ab * v + ac * w


@df.kernel
def eval_triangles_contact(
                                       # idx : df.tensor(int), # list of indices for colliding particles
    num_particles: int,                # size of particles
    x: df.tensor(df.float3),
    v: df.tensor(df.float3),
    indices: df.tensor(int),
    pose: df.tensor(df.mat22),
    activation: df.tensor(float),
    k_mu: float,
    k_lambda: float,
    k_damp: float,
    k_drag: float,
    k_lift: float,
    f: df.tensor(df.float3)):

    tid = df.tid()
    face_no = tid // num_particles     # which face
    particle_no = tid % num_particles  # which particle

    # index = df.load(idx, tid)
    pos = df.load(x, particle_no)      # at the moment, just one particle
                                       # vel0 = df.load(v, 0)

    i = df.load(indices, face_no * 3 + 0)
    j = df.load(indices, face_no * 3 + 1)
    k = df.load(indices, face_no * 3 + 2)

    if (i == particle_no or j == particle_no or k == particle_no):
        return

    p = df.load(x, i)        # point zero
    q = df.load(x, j)        # point one
    r = df.load(x, k)        # point two

    # vp = df.load(v, i) # vel zero
    # vq = df.load(v, j) # vel one
    # vr = df.load(v, k)  # vel two

    # qp = q-p # barycentric coordinates (centered at p)
    # rp = r-p

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest
    dist = df.dot(diff, diff)
    n = df.normalize(diff)
    c = df.min(dist - 0.01, 0.0)       # 0 unless within 0.01 of surface
                                       #c = df.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)
    fn = n * c * 1e5

    df.atomic_sub(f, particle_no, fn)

    # # apply forces (could do - f / 3 here)
    df.atomic_add(f, i, fn * bary[0])
    df.atomic_add(f, j, fn * bary[1])
    df.atomic_add(f, k, fn * bary[2])


@df.kernel
def eval_triangles_rigid_contacts(
    num_particles: int,                          # number of particles (size of contact_point)
    x: df.tensor(df.float3),                     # position of particles
    v: df.tensor(df.float3),
    indices: df.tensor(int),                     # triangle indices
    rigid_x: df.tensor(df.float3),               # rigid body positions
    rigid_r: df.tensor(df.quat),
    rigid_v: df.tensor(df.float3),
    rigid_w: df.tensor(df.float3),
    contact_body: df.tensor(int),
    contact_point: df.tensor(df.float3),         # position of contact points relative to body
    contact_dist: df.tensor(float),
    contact_mat: df.tensor(int),
    materials: df.tensor(float),
                                                 #   rigid_f : df.tensor(df.float3),
                                                 #   rigid_t : df.tensor(df.float3),
    tri_f: df.tensor(df.float3)):

    tid = df.tid()

    face_no = tid // num_particles     # which face
    particle_no = tid % num_particles  # which particle

    # -----------------------
    # load rigid body point
    c_body = df.load(contact_body, particle_no)
    c_point = df.load(contact_point, particle_no)
    c_dist = df.load(contact_dist, particle_no)
    c_mat = df.load(contact_mat, particle_no)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = df.load(materials, c_mat * 4 + 0)       # restitution coefficient
    kd = df.load(materials, c_mat * 4 + 1)       # damping coefficient
    kf = df.load(materials, c_mat * 4 + 2)       # friction coefficient
    mu = df.load(materials, c_mat * 4 + 3)       # coulomb friction

    x0 = df.load(rigid_x, c_body)      # position of colliding body
    r0 = df.load(rigid_r, c_body)      # orientation of colliding body

    v0 = df.load(rigid_v, c_body)
    w0 = df.load(rigid_w, c_body)

    # transform point to world space
    pos = x0 + df.rotate(r0, c_point)
    # use x0 as center, everything is offset from center of mass

    # moment arm
    r = pos - x0                       # basically just c_point in the new coordinates
    rhat = df.normalize(r)
    pos = pos + rhat * c_dist          # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    # contact point velocity
    dpdt = v0 + df.cross(w0, r)        # this is rigid velocity cross offset, so it's the velocity of the contact point.

    # -----------------------
    # load triangle
    i = df.load(indices, face_no * 3 + 0)
    j = df.load(indices, face_no * 3 + 1)
    k = df.load(indices, face_no * 3 + 2)

    p = df.load(x, i)        # point zero
    q = df.load(x, j)        # point one
    r = df.load(x, k)        # point two

    vp = df.load(v, i)       # vel zero
    vq = df.load(v, j)       # vel one
    vr = df.load(v, k)       # vel two

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest               # vector from tri to point
    dist = df.dot(diff, diff)          # squared distance
    n = df.normalize(diff)             # points into the object
    c = df.min(dist - 0.05, 0.0)       # 0 unless within 0.05 of surface
                                       #c = df.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)
                                       # fn = n * c * 1e6    # points towards cloth (both n and c are negative)

    # df.atomic_sub(tri_f, particle_no, fn)

    fn = c * ke    # normal force (restitution coefficient * how far inside for ground) (negative)

    vtri = vp * bary[0] + vq * bary[1] + vr * bary[2]         # bad approximation for centroid velocity
    vrel = vtri - dpdt

    vn = dot(n, vrel)        # velocity component of rigid in negative normal direction
    vt = vrel - n * vn       # velocity component not in normal direction

    # contact damping
    fd = 0.0 - df.max(vn, 0.0) * kd * df.step(c)           # again, negative, into the ground

    # # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)
    upper = 0.0 - lower      # workaround because no unary ops yet

    nx = cross(n, float3(0.0, 0.0, 1.0))         # basis vectors for tangent
    nz = cross(n, float3(1.0, 0.0, 0.0))

    vx = df.clamp(dot(nx * kf, vt), lower, upper)
    vz = df.clamp(dot(nz * kf, vt), lower, upper)

    ft = (nx * vx + nz * vz) * (0.0 - df.step(c))          # df.float3(vx, 0.0, vz)*df.step(c)

    # # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    # #ft = df.normalize(vt)*df.min(kf*df.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft

    df.atomic_add(tri_f, i, f_total * bary[0])
    df.atomic_add(tri_f, j, f_total * bary[1])
    df.atomic_add(tri_f, k, f_total * bary[2])


@df.kernel
def eval_bending(
    x: df.tensor(df.float3), v: df.tensor(df.float3), indices: df.tensor(int), rest: df.tensor(float), ke: float, kd: float, f: df.tensor(df.float3)):

    tid = df.tid()

    i = df.load(indices, tid * 4 + 0)
    j = df.load(indices, tid * 4 + 1)
    k = df.load(indices, tid * 4 + 2)
    l = df.load(indices, tid * 4 + 3)

    rest_angle = df.load(rest, tid)

    x1 = df.load(x, i)
    x2 = df.load(x, j)
    x3 = df.load(x, k)
    x4 = df.load(x, l)

    v1 = df.load(v, i)
    v2 = df.load(v, j)
    v3 = df.load(v, k)
    v4 = df.load(v, l)

    n1 = df.cross(x3 - x1, x4 - x1)    # normal to face 1
    n2 = df.cross(x4 - x2, x3 - x2)    # normal to face 2

    n1_length = df.length(n1)
    n2_length = df.length(n2)

    rcp_n1 = 1.0 / n1_length
    rcp_n2 = 1.0 / n2_length

    cos_theta = df.dot(n1, n2) * rcp_n1 * rcp_n2

    n1 = n1 * rcp_n1 * rcp_n1
    n2 = n2 * rcp_n2 * rcp_n2

    e = x4 - x3
    e_hat = df.normalize(e)
    e_length = df.length(e)

    s = df.sign(df.dot(df.cross(n2, n1), e_hat))
    angle = df.acos(cos_theta) * s

    d1 = n1 * e_length
    d2 = n2 * e_length
    d3 = n1 * df.dot(x1 - x4, e_hat) + n2 * df.dot(x2 - x4, e_hat)
    d4 = n1 * df.dot(x3 - x1, e_hat) + n2 * df.dot(x3 - x2, e_hat)

    # elastic
    f_elastic = ke * (angle - rest_angle)

    # damping
    f_damp = kd * (df.dot(d1, v1) + df.dot(d2, v2) + df.dot(d3, v3) + df.dot(d4, v4))

    # total force, proportional to edge length
    f_total = 0.0 - e_length * (f_elastic + f_damp)

    df.atomic_add(f, i, d1 * f_total)
    df.atomic_add(f, j, d2 * f_total)
    df.atomic_add(f, k, d3 * f_total)
    df.atomic_add(f, l, d4 * f_total)


@df.kernel
def eval_tetrahedra(x: df.tensor(df.float3),
                    v: df.tensor(df.float3),
                    indices: df.tensor(int),
                    pose: df.tensor(df.mat33),
                    activation: df.tensor(float),
                    materials: df.tensor(float),
                    f: df.tensor(df.float3)):

    tid = df.tid()

    i = df.load(indices, tid * 4 + 0)
    j = df.load(indices, tid * 4 + 1)
    k = df.load(indices, tid * 4 + 2)
    l = df.load(indices, tid * 4 + 3)

    act = df.load(activation, tid)

    k_mu = df.load(materials, tid * 3 + 0)
    k_lambda = df.load(materials, tid * 3 + 1)
    k_damp = df.load(materials, tid * 3 + 2)

    x0 = df.load(x, i)
    x1 = df.load(x, j)
    x2 = df.load(x, k)
    x3 = df.load(x, l)

    v0 = df.load(v, i)
    v1 = df.load(v, j)
    v2 = df.load(v, k)
    v3 = df.load(v, l)

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = df.mat33(x10, x20, x30)
    Dm = df.load(pose, tid)

    inv_rest_volume = df.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume
    k_damp = k_damp * rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm
    dFdt = df.mat33(v10, v20, v30) * Dm

    #-----------------------------
    # Neo-Hookean (with rest stability [Smith et al 2018])
    Ic = F[0, 0] * F[0, 0] + F[1, 1] * F[1, 1] + F[2, 2] * F[2, 2]

    # deviatoric part
    P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp
    H = P * df.transpose(Dm)

    f1 = df.float3(H[0, 0], H[1, 0], H[2, 0])
    f2 = df.float3(H[0, 1], H[1, 1], H[2, 1])
    f3 = df.float3(H[0, 2], H[1, 2], H[2, 2])

    # hydrostatic part
    J = df.determinant(F)

    #print(J)
    s = inv_rest_volume / 6.0
    dJdx1 = df.cross(x20, x30) * s
    dJdx2 = df.cross(x30, x10) * s
    dJdx3 = df.cross(x10, x20) * s

    f_volume = (J - alpha + act) * k_lambda
    f_damp = (df.dot(dJdx1, v1) + df.dot(dJdx2, v2) + df.dot(dJdx3, v3)) * k_damp

    f_total = f_volume + f_damp

    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    f3 = f3 + dJdx3 * f_total
    f0 = (f1 + f2 + f3) * (0.0 - 1.0)

    # apply forces
    df.atomic_sub(f, i, f0)
    df.atomic_sub(f, j, f1)
    df.atomic_sub(f, k, f2)
    df.atomic_sub(f, l, f3)


@df.kernel
def eval_contacts(x: df.tensor(df.float3), v: df.tensor(df.float3), ke: float, kd: float, kf: float, mu: float, f: df.tensor(df.float3)):

    tid = df.tid()           # this just handles contact of particles with the ground plane, nothing else.

    x0 = df.load(x, tid)
    v0 = df.load(v, tid)

    n = float3(0.0, 1.0, 0.0)          # why is the normal always y? Ground is always (0, 1, 0) normal

    c = df.min(dot(n, x0) - 0.01, 0.0)           # 0 unless within 0.01 of surface
                                                 #c = df.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)

    vn = dot(n, v0)
    vt = v0 - n * vn

    fn = n * c * ke

    # contact damping
    fd = n * df.min(vn, 0.0) * kd

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * c * ke
    upper = 0.0 - lower

    vx = clamp(dot(float3(kf, 0.0, 0.0), vt), lower, upper)
    vz = clamp(dot(float3(0.0, 0.0, kf), vt), lower, upper)

    ft = df.float3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = df.normalize(vt)*df.min(kf*df.length(vt), 0.0 - mu*c*ke)

    ftotal = fn + (fd + ft) * df.step(c)

    df.atomic_sub(f, tid, ftotal)


@df.kernel
def eval_rigid_contacts(rigid_x: df.tensor(df.float3),
                        rigid_r: df.tensor(df.quat),
                        rigid_v: df.tensor(df.float3),
                        rigid_w: df.tensor(df.float3),
                        contact_body: df.tensor(int),
                        contact_point: df.tensor(df.float3),
                        contact_dist: df.tensor(float),
                        contact_mat: df.tensor(int),
                        materials: df.tensor(float),
                        rigid_f: df.tensor(df.float3),
                        rigid_t: df.tensor(df.float3)):

    tid = df.tid()

    c_body = df.load(contact_body, tid)
    c_point = df.load(contact_point, tid)
    c_dist = df.load(contact_dist, tid)
    c_mat = df.load(contact_mat, tid)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = df.load(materials, c_mat * 4 + 0)       # restitution coefficient
    kd = df.load(materials, c_mat * 4 + 1)       # damping coefficient
    kf = df.load(materials, c_mat * 4 + 2)       # friction coefficient
    mu = df.load(materials, c_mat * 4 + 3)       # coulomb friction

    x0 = df.load(rigid_x, c_body)      # position of colliding body
    r0 = df.load(rigid_r, c_body)      # orientation of colliding body

    v0 = df.load(rigid_v, c_body)
    w0 = df.load(rigid_w, c_body)

    n = float3(0.0, 1.0, 0.0)

    # transform point to world space
    p = x0 + df.rotate(r0, c_point) - n * c_dist           # add on 'thickness' of shape, e.g.: radius of sphere/capsule
                                                           # use x0 as center, everything is offset from center of mass

    # moment arm
    r = p - x0     # basically just c_point in the new coordinates

    # contact point velocity
    dpdt = v0 + df.cross(w0, r)        # this is rigid velocity cross offset, so it's the velocity of the contact point.

    # check ground contact
    c = df.min(dot(n, p), 0.0)         # check if we're inside the ground

    vn = dot(n, dpdt)        # velocity component out of the ground
    vt = dpdt - n * vn       # velocity component not into the ground

    fn = c * ke    # nprmal force (restitution coefficient * how far inside for ground)

    # contact damping
    fd = df.min(vn, 0.0) * kd * df.step(c)       # again, velocity into the ground, negative

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)   # negative
    upper = 0.0 - lower      # positive, workaround for no unary ops

    vx = df.clamp(dot(float3(kf, 0.0, 0.0), vt), lower, upper)
    vz = df.clamp(dot(float3(0.0, 0.0, kf), vt), lower, upper)

    ft = df.float3(vx, 0.0, vz) * df.step(c)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = df.normalize(vt)*df.min(kf*df.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft
    t_total = df.cross(r, f_total)

    df.atomic_sub(rigid_f, c_body, f_total)
    df.atomic_sub(rigid_t, c_body, t_total)


# compute transform across a joint
@df.func
def jcalc_transform(type: int, axis: df.float3, joint_q: df.tensor(float), start: int):

    # prismatic
    if (type == 0):

        q = df.load(joint_q, start)
        X_jc = spatial_transform(axis * q, quat_identity())
        return X_jc

    # revolute
    if (type == 1):

        q = df.load(joint_q, start)
        X_jc = spatial_transform(float3(0.0, 0.0, 0.0), quat_from_axis_angle(axis, q))
        return X_jc

    # fixed
    if (type == 2):

        X_jc = spatial_transform_identity()
        return X_jc

    # free
    if (type == 3):

        px = df.load(joint_q, start + 0)
        py = df.load(joint_q, start + 1)
        pz = df.load(joint_q, start + 2)

        qx = df.load(joint_q, start + 3)
        qy = df.load(joint_q, start + 4)
        qz = df.load(joint_q, start + 5)
        qw = df.load(joint_q, start + 6)

        X_jc = spatial_transform(float3(px, py, pz), quat(qx, qy, qz, qw))
        return X_jc


# compute motion subspace and velocity for a joint
@df.func
def jcalc_motion(type: int, axis: df.float3, X_sc: df.spatial_transform, joint_S_s: df.tensor(df.spatial_vector), joint_qd: df.tensor(float), joint_start: int):

    # prismatic
    if (type == 0):

        S_s = df.spatial_transform_twist(X_sc, spatial_vector(float3(0.0, 0.0, 0.0), axis))
        v_j_s = S_s * df.load(joint_qd, joint_start)

        df.store(joint_S_s, joint_start, S_s)
        return v_j_s

    # revolute
    if (type == 1):

        S_s = df.spatial_transform_twist(X_sc, spatial_vector(axis, float3(0.0, 0.0, 0.0)))
        v_j_s = S_s * df.load(joint_qd, joint_start)

        df.store(joint_S_s, joint_start, S_s)
        return v_j_s

    # fixed
    if (type == 2):
        return spatial_vector()

    # free
    if (type == 3):

        v_j_c = spatial_vector(df.load(joint_qd, joint_start + 0),
                               df.load(joint_qd, joint_start + 1),
                               df.load(joint_qd, joint_start + 2),
                               df.load(joint_qd, joint_start + 3),
                               df.load(joint_qd, joint_start + 4),
                               df.load(joint_qd, joint_start + 5))

        v_j_s = spatial_transform_twist(X_sc, v_j_c)

        # write motion subspace
        df.store(joint_S_s, joint_start + 0, spatial_transform_twist(X_sc, spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        df.store(joint_S_s, joint_start + 1, spatial_transform_twist(X_sc, spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)))
        df.store(joint_S_s, joint_start + 2, spatial_transform_twist(X_sc, spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)))
        df.store(joint_S_s, joint_start + 3, spatial_transform_twist(X_sc, spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)))
        df.store(joint_S_s, joint_start + 4, spatial_transform_twist(X_sc, spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)))
        df.store(joint_S_s, joint_start + 5, spatial_transform_twist(X_sc, spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))

        return v_j_s


# # compute the velocity across a joint
# #@df.func
# def jcalc_velocity(self, type, S_s, joint_qd, start):

#     # prismatic
#     if (type == 0):
#         v_j_s = df.load(S_s, start)*df.load(joint_qd, start)
#         return v_j_s

#     # revolute
#     if (type == 1):
#         v_j_s = df.load(S_s, start)*df.load(joint_qd, start)
#         return v_j_s

#     # fixed
#     if (type == 2):
#         v_j_s = spatial_vector()
#         return v_j_s

#     # free
#     if (type == 3):
#         v_j_s =  S_s[start+0]*joint_qd[start+0]
#         v_j_s += S_s[start+1]*joint_qd[start+1]
#         v_j_s += S_s[start+2]*joint_qd[start+2]
#         v_j_s += S_s[start+3]*joint_qd[start+3]
#         v_j_s += S_s[start+4]*joint_qd[start+4]
#         v_j_s += S_s[start+5]*joint_qd[start+5]
#         return v_j_s


# computes joint space forces/torques in tau
@df.func
def jcalc_tau(type: int, joint_S_s: df.tensor(spatial_vector), joint_start: int, body_f_s: spatial_vector, tau: df.tensor(float)):

    # prismatic / revolute
    if (type == 0 or type == 1):
        S_s = df.load(joint_S_s, joint_start)
        df.store(tau, joint_start, spatial_dot(S_s, body_f_s))

    # free
    if (type == 3):
            
        for i in range(0, 6):
            S_s = df.load(joint_S_s, joint_start+i)
            df.store(tau, joint_start+i, spatial_dot(S_s, body_f_s))

    return 0


@df.func
def jcalc_integrate(type: int, joint_q: df.tensor(float), joint_qd: df.tensor(float), joint_qdd: df.tensor(float), coord_start: int, dof_start: int, dt: float):

    # prismatic / revolute
    if (type == 0 or type == 1):

        qdd = df.load(joint_qdd, dof_start)
        qd = df.load(joint_qd, dof_start)
        q = df.load(joint_q, coord_start)

        df.store(joint_qd, dof_start, qd + qdd*dt)
        df.store(joint_qd, coord_start, q + qd*dt)

    # free
    if (type == 3):

        # linear part
        for i in range(0, 3):
            
            qdd = df.load(joint_qdd, dof_start+3+i)
            qd = df.load(joint_qd, dof_start+3+i)
            q = df.load(joint_q, coord_start+i)

            df.store(joint_qd, dof_start+3+i, qd + qdd*dt)
            df.store(joint_q, coord_start+i, q + qd*dt)

        # angular part
        
        w = float3(df.load(joint_qd, dof_start + 0),
                   df.load(joint_qd, dof_start + 1),
                   df.load(joint_qd, dof_start + 2))

        # # quat and quat derivative
        # r = quat(
            
        #     q[q_start + 3], q[q_start + 4], q[q_start + 5], q[q_start + 6])
        # drdt = quat_multiply((*w, 0.0), r) * 0.5

        # # new orientation (normalized)
        # r_new = normalize(r + drdt * dt)

        # q[q_start + 3] = r_new[0]
        # q[q_start + 4] = r_new[1]
        # q[q_start + 5] = r_new[2]
        # q[q_start + 6] = r_new[3]

    return 0

@df.func
def compute_link_transform(i: int,
                           joint_type: df.tensor(int),
                           joint_parent: df.tensor(int),
                           joint_q_start: df.tensor(int),
                           joint_qd_start: df.tensor(int),
                           joint_q: df.tensor(float),
                           joint_X_pj: df.tensor(df.spatial_transform),
                           joint_X_cm: df.tensor(df.spatial_transform),
                           joint_axis: df.tensor(df.float3),
                           joint_S_s: df.tensor(df.spatial_vector),
                           body_X_sc: df.tensor(df.spatial_transform),
                           body_X_sm: df.tensor(df.spatial_transform)):

    # parent transform
    parent = load(joint_parent, i)

    # parent transform in spatial coordinates
    X_sp = spatial_transform_identity()
    if (parent >= 0):
        X_sp = load(body_X_sc, parent)

    type = load(joint_type, i)
    axis = load(joint_axis, i)
    coord_start = load(joint_q_start, i)
    dof_start = load(joint_qd_start, i)

    # compute transform across joint
    #X_jc = spatial_jcalc(type, joint_q, axis, coord_start)
    X_jc = jcalc_transform(type, axis, joint_q, coord_start)

    X_pj = load(joint_X_pj, i)
    X_sc = spatial_transform_multiply(X_sp, spatial_transform_multiply(X_pj, X_jc))

    # compute transform of center of mass
    X_cm = load(joint_X_cm, i)
    X_sm = spatial_transform_multiply(X_sc, X_cm)

    # compute motion subspace in space frame (J)
    #self.joint_S_s[i] = transform_twist(X_sc, S_c)

    # store geometry transforms
    store(body_X_sc, i, X_sc)
    store(body_X_sm, i, X_sm)

    return 0


@df.kernel
def eval_rigid_fk(articulation_start: df.tensor(int),
                  articulation_end: df.tensor(int),
                  joint_type: df.tensor(int),
                  joint_parent: df.tensor(int),
                  joint_q_start: df.tensor(int),
                  joint_qd_start: df.tensor(int),
                  joint_q: df.tensor(float),
                  joint_X_pj: df.tensor(df.spatial_transform),
                  joint_X_cm: df.tensor(df.spatial_transform),
                  joint_axis: df.tensor(df.float3),
                  joint_S_s: df.tensor(df.spatial_vector),
                  body_X_sc: df.tensor(df.spatial_transform),
                  body_X_sm: df.tensor(df.spatial_transform)):

    # one thread per-articulation
    index = tid()

    start = df.load(articulation_start, index)
    end = df.load(articulation_end, index)

    for i in range(start, end):
        compute_link_transform(i,
                               joint_type,
                               joint_parent,
                               joint_q_start,
                               joint_qd_start,
                               joint_q,
                               joint_X_pj,
                               joint_X_cm,
                               joint_axis,
                               joint_S_s,
                               body_X_sc,
                               body_X_sm)




#@df.func
def compute_link_velocity(i: int,
                          joint_type: df.tensor(int),
                          joint_parent: df.tensor(int),
                          joint_qd_start: df.tensor(int),
                          joint_qd: df.tensor(float),
                          joint_X_pj: df.tensor(df.spatial_transform),
                          joint_X_cm: df.tensor(df.spatial_transform),
                          joint_axis: df.tensor(df.float3),
                          body_I_m: df.tensor(df.spatial_matrix),
                          body_X_sc: df.tensor(df.spatial_transform),
                          body_X_sm: df.tensor(df.spatial_transform),
                          gravity: df.tensor(df.float3),
                          # outputs
                          joint_S_s: df.tensor(df.spatial_vector),
                          body_I_s: df.tensor(df.spatial_matrix),
                          body_v_s: df.tensor(df.spatial_vector),
                          body_f_s: df.tensor(df.spatial_vector),
                          body_a_s: df.tensor(df.spatial_vector)):

    type = df.load(joint_type, i)
    axis = df.load(joint_axis, i)
    dof_start = df.load(joint_qd_start, i)

    X_sc = df.load(body_X_sc, i)

    # compute motion subspace and velocity across the joint (stores S_s to global memory)
    v_j_s = jcalc_motion(type, axis, X_sc, joint_S_s, joint_qd, dof_start)

    # parent velocity
    parent = df.load(joint_parent, i)

    v_parent_s = spatial_vector()
    a_parent_s = spatial_vector()

    if (parent >= 0):
        v_parent_s = df.load(body_v_s, parent)
        a_parent_s = df.load(body_a_s, parent)

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s) # + self.joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = df.load(body_X_sm, i)
    I_m = df.load(body_I_m, i)

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    g = df.load(gravity, 0)

    m = I_m[3, 3]
    f_ext_m = spatial_vector(float3(), g) * m
    f_ext_s = f_ext_m # todo: spatial_transform_wrench(X_sm, f_ext_m)

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = df.mul(I_s, a_s) + spatial_cross_dual(v_s, df.mul(I_s, v_s))

    df.store(body_v_s, i, v_s)
    df.store(body_a_s, i, a_s)
    df.store(body_f_s, i, f_b_s - f_ext_s)
    df.store(body_I_s, i, I_s)

    return 0


#@df.func
def compute_link_tau(i: int,
                     joint_type: df.tensor(int),
                     joint_parent: df.tensor(int),
                     joint_q_start: df.tensor(int),
                     joint_qd_start: df.tensor(int),
                     joint_S_s: df.tensor(df.spatial_vector),
                     body_f_s: df.tensor(df.spatial_vector),
                     body_f_subtree_s: df.tensor(df.spatial_vector),
                     tau: df.tensor(float)):

    type = df.load(joint_type, i)
    parent = df.load(joint_parent, i)
    dof_start = df.load(joint_qd_start, i)

    f_s = df.load(body_f_s, i)                   # external forces on body
    f_child = df.load(body_f_subtree_s, i)       # forces acting on subtree of body

    f_tot = f_s + f_child

    # compute joint-space forces
    jcalc_tau(type, joint_S_s, dof_start, f_s, tau)

    # update parent forces
    df.atomic_add(body_f_subtree_s, parent, f_tot)

    return 0


#@df.kernel
def eval_rigid_id(joint_type: df.tensor(int),
                  joint_parent: df.tensor(int),
                  joint_q_start: df.tensor(int),
                  joint_qd_start: df.tensor(int),
                  joint_q: df.tensor(float),
                  joint_X_pj: df.tensor(df.spatial_transform),
                  joint_X_cm: df.tensor(df.spatial_transform),
                  joint_axis: df.tensor(df.float3),
                  joint_S_s: df.tensor(df.spatial_vector),
                  body_X_sc: df.tensor(df.spatial_transform),
                  body_X_sm: df.tensor(df.spatial_transform)):

    # one thread per-articulation
    index = tid()

    start = df.load(articulation_start, index)
    end = df.load(articulation_end, index)

    for i in range(0, count):
        compute_link_velocity(i,
                              joint_type,
                              joint_parent,
                              joint_q_start,
                              joint_qd_start,
                              joint_q,
                              joint_X_pj,
                              joint_X_cm,
                              joint_axis,
                              joint_S_s,
                              body_X_sc,
                              body_X_sm)

    for i in range(count - 1, -1):
        compute_link_tau(
            i,
            joint_type,
            joint_parent,
            joint_q_start,
            joint_qd_start,
        )




# define PyTorch autograd op to wrap simulate func
class SimulateFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, integrator, model, state_in, state_out, dt, *tensors):

        # record launches
        ctx.tape = df.Tape()
        ctx.inputs = tensors
        ctx.outputs = df.to_weak_list(state_out.flatten())

        # simulate
        integrator.simulate(ctx.tape, model, state_in, state_out, dt)

        return tuple(state_out.flatten())

    @staticmethod
    def backward(ctx, *grads):

        # ensure grads are contiguous in memory
        adj_outputs = df.make_contiguous(grads)

        # register outputs with tape
        outputs = df.to_strong_list(ctx.outputs)        
        for o in range(len(outputs)):
            ctx.tape.adjoints[outputs[o]] = adj_outputs[o]

        # replay launches backwards
        ctx.tape.replay()

        # find adjoint of inputs
        adj_inputs = []
        for i in ctx.inputs:

            if i in ctx.tape.adjoints:
                adj_inputs.append(ctx.tape.adjoints[i])
            else:
                adj_inputs.append(None)

        # free the tape
        ctx.tape.reset()

        # filter grads to replace empty tensors / no grad / constant params with None
        return (None, None, None, None, None, *df.filter_grads(adj_inputs))


class SemiImplicitIntegrator:

    def __init__(self):
        pass

    def simulate(self, tape, model, state_in, state_out, dt):
        # if config.use_taichi:
        #     from dflex.taichi_sim import TaichiSemiImplicitIntegrator
        #     return TaichiSemiImplicitIntegrator.forward(model, state_in, state_out, dt)

        with util.ScopedTimer("simulate", False):

            # alloc particle force buffer
            if (model.particle_count):
                f_particle = torch.zeros_like(state_in.u, requires_grad=True)

            # alloc rigid force buffer
            if (model.rigid_count):
                f_rigid = torch.zeros_like(state_in.rigid_v, requires_grad=True)
                t_rigid = torch.zeros_like(state_in.rigid_w, requires_grad=True)

            # damped springs
            if (model.spring_count):

                tape.launch(func=eval_springs,
                            dim=model.spring_count,
                            inputs=[state_in.q, state_in.u, model.spring_indices, model.spring_rest_length, model.spring_stiffness, model.spring_damping],
                            outputs=[f_particle],
                            adapter=model.adapter)

            # triangle elastic and lift/drag forces
            if (model.tri_count and model.tri_ke > 0.0):

                tape.launch(func=eval_triangles,
                            dim=model.tri_count,
                            inputs=[
                                state_in.q,
                                state_in.u,
                                model.tri_indices,
                                model.tri_poses,
                                model.tri_activations,
                                model.tri_ke,
                                model.tri_ka,
                                model.tri_kd,
                                model.tri_drag,
                                model.tri_lift
                            ],
                            outputs=[f_particle],
                            adapter=model.adapter)

            # triangle/triangle contacts
            if (model.tri_collisions and model.tri_count and model.tri_ke > 0.0):
                tape.launch(func=eval_triangles_contact,
                            dim=model.tri_count * model.particle_count,
                            inputs=[
                                model.particle_count,
                                state_in.q,
                                state_in.u,
                                model.tri_indices,
                                model.tri_poses,
                                model.tri_activations,
                                model.tri_ke,
                                model.tri_ka,
                                model.tri_kd,
                                model.tri_drag,
                                model.tri_lift
                            ],
                            outputs=[f_particle],
                            adapter=model.adapter)

            # triangle bending
            if (model.edge_count):

                tape.launch(func=eval_bending,
                            dim=model.edge_count,
                            inputs=[state_in.q, state_in.u, model.edge_indices, model.edge_rest_angle, model.edge_ke, model.edge_kd],
                            outputs=[f_particle],
                            adapter=model.adapter)

            # ground contact
            if (model.ground):

                tape.launch(func=eval_contacts,
                            dim=model.particle_count,
                            inputs=[state_in.q, state_in.u, model.contact_ke, model.contact_kd, model.contact_kf, model.contact_mu],
                            outputs=[f_particle],
                            adapter=model.adapter)

            # tetrahedral FEM
            if (model.tet_count):

                tape.launch(func=eval_tetrahedra,
                            dim=model.tet_count,
                            inputs=[state_in.q, state_in.u, model.tet_indices, model.tet_poses, model.tet_activations, model.tet_materials],
                            outputs=[f_particle],
                            adapter=model.adapter)

            #----------------------------
            # rigid forces

            # eval contacts with all shapes that have been specified.
            if (model.contact_count):

                tape.launch(func=eval_rigid_contacts,
                            dim=model.contact_count,
                            inputs=[
                                state_in.rigid_x,
                                state_in.rigid_r,
                                state_in.rigid_v,
                                state_in.rigid_w,
                                model.contact_body0,
                                model.contact_point0,
                                model.contact_dist,
                                model.contact_material,
                                model.shape_materials
                            ],
                            outputs=[f_rigid, t_rigid],
                            adapter=model.adapter)

                # todo: combine this with the kernel above
                if model.tri_collisions:
                    tape.launch(func=eval_triangles_rigid_contacts,
                                dim=model.contact_count * model.tri_count,
                                inputs=[
                                    model.contact_count,
                                    state_in.q,
                                    state_in.u,
                                    model.tri_indices,
                                    state_in.rigid_x,
                                    state_in.rigid_r,
                                    state_in.rigid_v,
                                    state_in.rigid_w,
                                    model.contact_body0,
                                    model.contact_point0,
                                    model.contact_dist,
                                    model.contact_material,
                                    model.shape_materials
                                ],
                                outputs=[f_particle],
                                adapter=model.adapter)

            #----------------------------
            # integrate

            if (model.particle_count):
                tape.launch(func=integrate_particles,
                            dim=model.particle_count,
                            inputs=[state_in.q, state_in.u, f_particle, model.particle_inv_mass, model.gravity, dt],
                            outputs=[state_out.q, state_out.u],
                            adapter=model.adapter)

            if (model.rigid_count):
                tape.launch(func=integrate_rigids,
                            dim=model.rigid_count,
                            inputs=[
                                state_in.rigid_x,
                                state_in.rigid_r,
                                state_in.rigid_v,
                                state_in.rigid_w,
                                f_rigid,
                                t_rigid,
                                model.rigid_inv_mass,
                                model.rigid_inv_inertia,
                                model.gravity.float(),
                                dt
                            ],
                            outputs=[state_out.rigid_x, state_out.rigid_r, state_out.rigid_v, state_out.rigid_w],
                            adapter=model.adapter)

            return state_out


    def forward(self, model, state_in, dt):

        if config.no_grad:

            # if no gradient required then do inplace update
            self.simulate(df.Tape(), model, state_in, state_in, dt)
            return state_in

        else:

            # get list of inputs and outputs for PyTorch tensor tracking            
            inputs = [*state_in.flatten(), *model.flatten()]

            # allocate new output
            state_out = state_in.clone()

            # run sim as a PyTorch op
            tensors = SimulateFunc.apply(self, model, state_in, state_out, dt, *inputs)

            return state_out
            

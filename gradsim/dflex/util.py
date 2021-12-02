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

import timeit
import math
import numpy as np
import gc
import torch
import cProfile

log_output = ""

def log(s):
    print(s)
    global log_output
    log_output = log_output + s + "\n"

# short hands


def length(a):
    return np.linalg.norm(a)


def length_sq(a):
    return np.dot(a, a)


# NumPy has no normalize() method..
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    return v / norm


def skew(v):

    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# math utils
def quat(i, j, k, w):
    return np.array([i, j, k, w])


def quat_identity():
    return np.array((0.0, 0.0, 0.0, 1.0))


def quat_inverse(q):
    return np.array((-q[0], -q[1], -q[2], q[3]))


def quat_from_axis_angle(axis, angle):
    v = np.array(axis)

    half = angle * 0.5
    w = math.cos(half)

    sin_theta_over_two = math.sin(half)
    v *= sin_theta_over_two

    return np.array((v[0], v[1], v[2], w))


# rotate a vector
def quat_rotate(q, x):
    axis = np.array((q[0], q[1], q[2]))
    return x * (2.0 * q[3] * q[3] - 1.0) + np.cross(axis, x) * q[3] * 2.0 + axis * np.dot(axis, x) * 2.0


# multiply two quats
def quat_multiply(a, b):

    return np.array((a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - b[1] * a[2],
                     a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - b[2] * a[0],
                     a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - b[0] * a[1],
                     a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]))


# convert to mat33
def quat_to_matrix(q):

    c1 = quat_rotate(q, np.array((1.0, 0.0, 0.0)))
    c2 = quat_rotate(q, np.array((0.0, 1.0, 0.0)))
    c3 = quat_rotate(q, np.array((0.0, 0.0, 1.0)))

    return np.array([c1, c2, c3]).T


def quat_rpy(roll, pitch, yaw):

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    w = (cy * cr * cp + sy * sr * sp)
    x = (cy * sr * cp - sy * cr * sp)
    y = (cy * cr * sp + sy * sr * cp)
    z = (sy * cr * cp - cy * sr * sp)

    return (x, y, z, w)


# rigid body transform


def transform(x, r):
    return (np.array(x), np.array(r))


def transform_identity():
    return (np.array((0.0, 0.0, 0.0)), quat_identity())


# se(3) -> SE(3), Park & Lynch pg. 105, screw in [w, v] normalized form
def transform_exp(s, angle):

    w = np.array(s[0:3])
    v = np.array(s[3:6])

    if (length(w) < 1.0):
        r = quat_identity()
    else:
        r = quat_from_axis_angle(w, angle)

    t = v * angle + (1.0 - math.cos(angle)) * np.cross(w, v) + (angle - math.sin(angle)) * np.cross(w, np.cross(w, v))

    return (t, r)


def transform_inverse(t):
    q_inv = quat_inverse(t[1])
    return (-quat_rotate(q_inv, t[0]), q_inv)


def transform_vector(t, v):
    return quat_rotate(t[1], v)


def transform_point(t, p):
    return np.array(t[0]) + quat_rotate(t[1], p)


def transform_multiply(t, u):
    return (quat_rotate(t[1], u[0]) + t[0], quat_multiply(t[1], u[1]))


# flatten an array of transforms (p,q) format to a 7-vector
def transform_flatten(t):
    return np.array([*t[0], *t[1]])


# expand a 7-vec to a tuple of arrays
def transform_expand(t):
    return (np.array(t[0:3]), np.array(t[3:7]))


# convert array of transforms to a array of 7-vecs
def transform_flatten_list(xforms):
    exp = lambda t: transform_flatten(t)
    return list(map(exp, xforms))


def transform_expand_list(xforms):
    exp = lambda t: transform_expand(t)
    return list(map(exp, xforms))


# spatial operators


# AdT
def spatial_adjoint(t):

    R = quat_to_matrix(t[1])
    w = skew(t[0])

    A = np.zeros((6, 6))
    A[0:3, 0:3] = R
    A[3:6, 0:3] = np.dot(w, R)
    A[3:6, 3:6] = R

    return A


# (AdT)^-T
def spatial_adjoint_dual(t):

    R = quat_to_matrix(t[1])
    w = skew(t[0])

    A = np.zeros((6, 6))
    A[0:3, 0:3] = R
    A[0:3, 3:6] = np.dot(w, R)
    A[3:6, 3:6] = R

    return A


# AdT*s
def transform_twist(t_ab, s_b):
    return np.dot(spatial_adjoint(t_ab), s_b)


# AdT^{-T}*s
def transform_wrench(t_ab, f_b):
    return np.dot(spatial_adjoint_dual(t_ab), f_b)


# transform inertia in b frame to a frame
def transform_inertia(t_ab, I_b):

    # todo: write specialized method
    I_a = np.dot(np.dot(spatial_adjoint_dual(t_ab), I_b), spatial_adjoint(transform_inverse(t_ab)))
    return I_a


def translate_twist(p_ab, s_b):
    w = s_b[0:3]
    v = np.cross(p_ab, s_b[0:3]) + s_b[3:6]

    return np.array((*w, *v))


def translate_wrench(p_ab, s_b):
    w = s_b[0:3] + np.cross(p_ab, s_b[3:6])
    v = s_b[3:6]

    return np.array((*w, *v))


def spatial_vector(v=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
    return np.array(v)


# ad_V pg. 289 L&P, pg. 25 Featherstone
def spatial_cross(a, b):
    w = np.cross(a[0:3], b[0:3])
    v = np.cross(a[3:6], b[0:3]) + np.cross(a[0:3], b[3:6])

    return np.array((*w, *v))


# ad_V^T pg. 290 L&P,  pg. 25 Featurestone, note this does not includes the sign flip in the definition
def spatial_cross_dual(a, b):
    w = np.cross(a[0:3], b[0:3]) + np.cross(a[3:6], b[3:6])
    v = np.cross(a[0:3], b[3:6])

    return np.array((*w, *v))


def spatial_dot(a, b):
    return np.dot(a, b)


def spatial_outer(a, b):
    return np.outer(a, b)


def spatial_matrix():
    return np.zeros((6, 6))


def spatial_matrix_from_inertia(I, m):

    G = spatial_matrix()

    G[0:3, 0:3] = I
    G[3, 3] = m
    G[4, 4] = m
    G[5, 5] = m

    return G


# solves x = I^(-1)b
def spatial_solve(I, b):
    return np.dot(np.linalg.inv(I), b)



# timer utils


class ScopedTimer:

    indent = -1

    enabled = True

    def __init__(self, name, active=True, detailed=False):
        self.name = name
        self.active = active and self.enabled
        self.detailed = detailed

    def __enter__(self):

        if (self.active):
            self.start = timeit.default_timer()
            ScopedTimer.indent += 1

            if (self.detailed):
                self.cp = cProfile.Profile()
                self.cp.clear()
                self.cp.enable()    


    def __exit__(self, exc_type, exc_value, traceback):

        if (self.active):
            elapsed = (timeit.default_timer() - self.start) * 1000.0

            indent = ""
            for i in range(ScopedTimer.indent):
                indent += "\t"

            log("{}{} took {:.2}ms".format(indent, self.name, elapsed))

            ScopedTimer.indent -= 1

            if (self.detailed):
                self.cp.disable()
                self.cp.print_stats(sort='tottime')


# code snippet for invoking cProfile
# cp = cProfile.Profile()
# cp.enable()
# for i in range(1000):
#     self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

# cp.disable()
# cp.print_stats(sort='tottime')
# exit(0)


# represent an edge between v0, v1 with connected faces f0, f1, and opposite vertex o0, and o1
# winding is such that first tri can be reconstructed as {v0, v1, o0}, and second tri as { v1, v0, o1 }
class MeshEdge:
    def __init__(self, v0, v1, o0, o1, f0, f1):
        self.v0 = v0         # vertex 0
        self.v1 = v1         # vertex 1
        self.o0 = o0         # opposite vertex 1
        self.o1 = o1         # opposite vertex 2
        self.f0 = f0         # index of tri1
        self.f1 = f1         # index of tri2


class MeshAdjacency:
    def __init__(self, indices, num_tris):

        # map edges (v0, v1) to faces (f0, f1)
        self.edges = {}
        self.indices = indices

        for index, tri in enumerate(indices):
            self.add_edge(tri[0], tri[1], tri[2], index)
            self.add_edge(tri[1], tri[2], tri[0], index)
            self.add_edge(tri[2], tri[0], tri[1], index)

    def add_edge(self, i0, i1, o, f):  # index1, index2, index3, index of triangle

        key = (min(i0, i1), max(i0, i1))
        edge = None

        if key in self.edges:

            edge = self.edges[key]

            if (edge.f1 != -1):
                print("Detected non-manifold edge")
                return
            else:
                # update other side of the edge
                edge.o1 = o
                edge.f1 = f
        else:
            # create new edge with opposite yet to be filled
            edge = MeshEdge(i0, i1, o, -1, f, -1)

        self.edges[key] = edge

    def opposite_vertex(self, edge):
        pass




def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            # print('%s\t\t%s\t\t%.2f' % (
            #     element_type,
            #     size,
            #     mem) )
        print('Type: %s Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (mem_type, total_numel, total_mem) )

    gc.collect()

    LEN = 65
    objects = gc.get_objects()
    #print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    #_mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)

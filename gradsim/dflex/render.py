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

from pxr import Usd, UsdGeom, Gf

import dflex.sim
import dflex.util

import math


def usd_add_xform(prim):

    t = prim.AddTranslateOp()
    r = prim.AddOrientOp()
    s = prim.AddScaleOp()


def usd_set_xform(xform, pos, rot, scale, time):

    xform_ops = xform.GetOrderedXformOps()

    xform_ops[0].Set(Gf.Vec3d(pos), time)
    xform_ops[1].Set(Gf.Quatf(rot[3], rot[0], rot[1], rot[2]), time)
    xform_ops[2].Set(Gf.Vec3d(scale), time)


class UsdRenderer:
    def __init__(self, model, stage):

        self.stage = stage
        self.model = model

        self.draw_points = True
        self.draw_springs = False
        self.draw_triangles = False

        if (stage.GetPrimAtPath("/root")):
            stage.RemovePrim("/root")

        self.root = UsdGeom.Xform.Define(stage, '/root')

        # add sphere instancer
        self.particle_instancer = UsdGeom.PointInstancer.Define(stage, self.root.GetPath().AppendChild("particle_instancer"))
        self.particle_instancer_sphere = UsdGeom.Sphere.Define(stage, self.particle_instancer.GetPath().AppendChild("sphere"))
        self.particle_instancer_sphere.GetRadiusAttr().Set(model.particle_radius)

        self.particle_instancer.CreatePrototypesRel().SetTargets([self.particle_instancer_sphere.GetPath()])
        self.particle_instancer.CreateProtoIndicesAttr().Set([0] * model.particle_count)

        # add line instancer
        if (self.model.spring_count > 0):
            self.line_instancer = UsdGeom.PointInstancer.Define(stage, self.root.GetPath().AppendChild("line_instancer"))
            self.line_instancer_cylinder = UsdGeom.Capsule.Define(stage, self.line_instancer.GetPath().AppendChild("cylinder"))
            self.line_instancer_cylinder.GetRadiusAttr().Set(0.01)

            self.line_instancer.CreatePrototypesRel().SetTargets([self.line_instancer_cylinder.GetPath()])
            self.line_instancer.CreateProtoIndicesAttr().Set([0] * model.spring_count)

        self.stage.SetDefaultPrim(self.root.GetPrim())

        # time codes
        try:
            self.stage.SetStartTimeCode(0.0)
            self.stage.SetEndTimeCode(0.0)
            self.stage.SetTimeCodesPerSecond(60.0)
        except:
            pass

        # add dynamic cloth mesh
        if (model.tri_count > 0):

            self.cloth_mesh = UsdGeom.Mesh.Define(stage, self.root.GetPath().AppendChild("cloth"))

            self.cloth_remap = {}
            self.cloth_verts = []
            self.cloth_indices = []

            # USD needs a contiguous vertex buffer, use a dict to map from simulation indices->render indices
            indices = self.model.tri_indices.flatten().tolist()

            for i in indices:

                if i not in self.cloth_remap:

                    # copy vertex
                    new_index = len(self.cloth_verts)

                    self.cloth_verts.append(self.model.particle_x[i].tolist())
                    self.cloth_indices.append(new_index)

                    self.cloth_remap[i] = new_index

                else:
                    self.cloth_indices.append(self.cloth_remap[i])

            self.cloth_mesh.GetPointsAttr().Set(self.cloth_verts)
            self.cloth_mesh.GetFaceVertexIndicesAttr().Set(self.cloth_indices)
            self.cloth_mesh.GetFaceVertexCountsAttr().Set([3] * model.tri_count)

        else:
            self.cloth_mesh = None

        # built-in ground plane
        if (model.ground):

            size = 10.0

            mesh = UsdGeom.Mesh.Define(stage, self.root.GetPath().AppendChild("plane_0"))
            mesh.CreateDoubleSidedAttr().Set(True)

            points = ((-size, 0.0, -size), (size, 0.0, -size), (size, 0.0, size), (-size, 0.0, size))
            normals = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
            counts = (4, )
            indices = [0, 1, 2, 3]

            mesh.GetPointsAttr().Set(points)
            mesh.GetNormalsAttr().Set(normals)
            mesh.GetFaceVertexCountsAttr().Set(counts)
            mesh.GetFaceVertexIndicesAttr().Set(indices)

        # add rigid bodies xform root
        for b in range(model.rigid_count):

            xform = UsdGeom.Xform.Define(stage, self.root.GetPath().AppendChild("body_" + str(b)))
            usd_add_xform(xform)

        # add rigid body shapes
        for s in range(model.shape_count):

            parent_path = self.root.GetPath()
            if model.shape_body[s] >= 0:
                parent_path = parent_path.AppendChild("body_" + str(model.shape_body[s].item()))

            geo_type = model.shape_geo_type[s].item()
            geo_scale = model.shape_geo_scale[s].tolist()
            geo_src = model.shape_geo_src[s]

            if (geo_type == dflex.sim.GEO_PLANE):

                # plane mesh
                size = 1000.0

                mesh = UsdGeom.Mesh.Define(stage, parent_path.AppendChild("plane_" + str(s)))
                mesh.CreateDoubleSidedAttr().Set(True)

                points = ((-size, 0.0, -size), (size, 0.0, -size), (size, 0.0, size), (-size, 0.0, size))
                normals = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
                counts = (4, )
                indices = [0, 1, 2, 3]

                mesh.GetPointsAttr().Set(points)
                mesh.GetNormalsAttr().Set(normals)
                mesh.GetFaceVertexCountsAttr().Set(counts)
                mesh.GetFaceVertexIndicesAttr().Set(indices)

            elif (geo_type == dflex.sim.GEO_SPHERE):

                mesh = UsdGeom.Sphere.Define(stage, parent_path.AppendChild("sphere_" + str(s)))
                mesh.GetRadiusAttr().Set(geo_scale[0])     #.item())

            elif (geo_type == dflex.sim.GEO_CAPSULE):
                mesh = UsdGeom.Capsule.Define(stage, parent_path.AppendChild("capsule_" + str(s)))
                mesh.GetRadiusAttr().Set(geo_scale[0])
                mesh.GetHeightAttr().Set(geo_scale[1] * 2.0)

                usd_add_xform(mesh)
                usd_set_xform(mesh, (0.0, 0.0, 0.0), dflex.util.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi * 0.5), (1.0, 1.0, 1.0), 0.0)

            elif (geo_type == dflex.sim.GEO_BOX):
                mesh = UsdGeom.Cube.Define(stage, parent_path.AppendChild("box_" + str(s)))
                #mesh.GetSizeAttr().Set((geo_scale[0], geo_scale[1], geo_scale[2]))

                usd_add_xform(mesh)
                usd_set_xform(mesh, (0.0, 0.0, 0.0), dflex.util.quat_identity(), (geo_scale[0], geo_scale[1], geo_scale[2]), 0.0)

            elif (geo_type == dflex.sim.GEO_MESH):

                mesh = UsdGeom.Mesh.Define(stage, parent_path.AppendChild("mesh_" + str(s)))
                mesh.GetPointsAttr().Set(geo_src.vertices)
                mesh.GetFaceVertexIndicesAttr().Set(geo_src.indices)
                mesh.GetFaceVertexCountsAttr().Set([3] * int(len(geo_src.indices) / 3))

                usd_add_xform(mesh)
                usd_set_xform(mesh, (0.0, 0.0, 0.0), dflex.util.quat_identity(), (geo_scale[0], geo_scale[1], geo_scale[2]), 0.0)

            elif (geo_type == dflex.sim.GEO_SDF):
                pass

    def update(self, state, time):
        # use 60 fps time codes
        time = time * 60.0

        try:
            self.stage.SetEndTimeCode(time)
        except:
            pass

        # convert to list
        particle_x = state.q.tolist()
        particle_orientations = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * self.model.particle_count

        self.particle_instancer.GetPositionsAttr().Set(particle_x, time)
        self.particle_instancer.GetOrientationsAttr().Set(particle_orientations, time)

        # update cloth
        if (self.cloth_mesh):

            for k, v in self.cloth_remap.items():
                self.cloth_verts[v] = particle_x[k]

            self.cloth_mesh.GetPointsAttr().Set(self.cloth_verts, time)

        # update springs
        if (self.model.spring_count > 0):

            line_positions = []
            line_rotations = []
            line_scales = []

            for i in range(self.model.spring_count):

                index0 = self.model.spring_indices[i * 2 + 0]
                index1 = self.model.spring_indices[i * 2 + 1]

                pos0 = particle_x[index0]
                pos1 = particle_x[index1]

                (pos, rot, scale) = self.compute_segment_xform(Gf.Vec3f(pos0), Gf.Vec3f(pos1))

                line_positions.append(pos)
                line_rotations.append(rot)
                line_scales.append(scale)

            self.line_instancer.GetPositionsAttr().Set(line_positions, time)
            self.line_instancer.GetOrientationsAttr().Set(line_rotations, time)
            self.line_instancer.GetScalesAttr().Set(line_scales, time)

        # rigids
        for b in range(self.model.rigid_count):

            #xform = UsdGeom.Xform.Define(self.stage, self.root.GetPath().AppendChild("body_" + str(b)))

            xform = UsdGeom.Xform(self.stage.GetPrimAtPath(self.root.GetPath().AppendChild("body_" + str(b))))

            x = state.rigid_x[b].tolist()
            r = state.rigid_r[b].tolist()

            usd_set_xform(xform, x, r, (1.0, 1.0, 1.0), time)

    def add_sphere(self, pos, radius, name):

        sphere = UsdGeom.Sphere.Define(self.stage, self.root.GetPath().AppendChild(name))
        sphere.GetRadiusAttr().Set(radius)

        mat = Gf.Matrix4d()
        mat.SetIdentity()
        mat.SetTranslateOnly(Gf.Vec3d(pos))

        op = sphere.MakeMatrixXform()
        op.Set(mat)

    # transforms a cylinder such that it connects the two points pos0, pos1
    def compute_segment_xform(self, pos0, pos1):

        mid = (pos0 + pos1) * 0.5
        height = (pos1 - pos0).GetLength()

        dir = (pos1 - pos0) / height

        rot = Gf.Rotation()
        rot.SetRotateInto((0.0, 0.0, 1.0), Gf.Vec3d(dir))

        scale = Gf.Vec3f(1.0, 1.0, height)

        return (mid, Gf.Quath(rot.GetQuat()), scale)

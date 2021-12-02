import glob
import numpy as np
import os

from gradsim.assets.primitives import Primitive, get_primitive_obj
from gradsim.urdf import fill_urdf, write_tmp_urdf
from gradsim.utils.defaults import Defaults
from gradsim.utils.misc import random_bright_color


# using this to send back samples
class ShapeSample(object):
    shape = None
    mass = None
    fric = None
    elas = None
    color = None
    scale = None
    obj = None  # path to OBJ
    urdf = None  # path to URDF

    def __str__(self):
        return (
            f"{self.shape} with \n"
            f"- mass {self.mass},\n"
            f"- friction {self.fric}\n"
            f"- elasticity {self.elas}\n"
            f"- color {self.color}\n"
            f"- scale {self.scale}\n"
            f"- using obj path '{self.obj}'\n"
            f"- using urdf path '{self.urdf}'"
        )


class PrimitiveGenerator(object):
    def __init__(
        self,
        default_shape=Defaults.SHAPE,
        random_shape=True,  # if False, set self.default_shape,
        random_mass=True,  # if False, set self.default_mass,
        random_fric=True,  # if False, set self.default_fric,
        random_elas=True,  # if False, set self.default_elas
        random_color=True,  # if False, set self.default_color
        random_scale=False,  # if False, set self.default_scale
        mass_scaling=1,
        mass_offset=0,
        fric_scaling=1,
        fric_offset=0,
        elas_scaling=1,
        elas_offset=0,
        scale_scaling=1,  # ha-haaaa
        scale_offset=0,
        # randomseed=42,
        data_type="primitive",
        data_dir="",
        # randomseed=42,
    ):
        super().__init__()

        # np.random.seed(randomseed)

        self.random_shape = random_shape
        self.default_shape = default_shape

        self.random_mass = random_mass
        self.default_mass = Defaults.MASS
        self.mass_scaling = mass_scaling
        self.mass_offset = mass_offset

        self.random_fric = random_fric
        self.default_fric = Defaults.FRICTION
        self.fric_scaling = fric_scaling
        self.fric_offset = fric_offset

        self.random_elas = random_elas
        self.default_elas = Defaults.RESTITUTION
        self.elas_scaling = elas_scaling
        self.elas_offset = elas_offset

        self.random_color = random_color
        self.default_color = Defaults.COLOR

        self.random_scale = random_scale
        self.default_scale = Defaults.SCALE
        self.scale_scaling = scale_scaling
        self.scale_offset = scale_offset

        self.data_type = data_type
        self.data_dir = data_dir
        if self.data_type == "shapenet":
            assert os.path.exists(data_dir), f"Must provide a valid data_dir if shapenet! Given {str(self.data_dir)}"
            self.obj_dirs = sorted(glob.glob(os.path.join(self.data_dir, "*/*")))
        elif self.data_type != 'primitive':
            assert os.path.exists(data_dir), f"Must provide a valid data_dir if not primitive! Given {str(self.data_dir)}"
            self.objs = sorted(glob.glob(os.path.join(self.data_dir, "*.obj")))

    def sample(self):
        out = ShapeSample()

        if self.data_type == "shapenet":
            obj_dir = np.random.choice(self.obj_dirs) if self.random_shape else self.obj_dirs[0]
            out.obj = os.path.join(obj_dir, 'model.obj')
            out.shape = self.obj_dirs.index(obj_dir)
        elif self.data_type == "primitive":
            out.shape = np.random.choice(list(Primitive)) if self.random_shape else self.default_shape
            out.obj = get_primitive_obj(out.shape)
        else:
            out.obj = np.random.choice(self.objs) if self.random_shape else self.objs[0]
            out.shape = self.objs.index(out.obj)

        out.mass = self.default_mass
        if self.random_mass:
            out.mass = np.random.rand() * self.mass_scaling + self.mass_offset

        out.fric = self.default_fric
        if self.random_fric:
            out.fric = np.random.rand() * self.fric_scaling + self.fric_offset

        out.elas = self.default_elas
        if self.random_elas:
            out.elas = np.random.rand() * self.elas_scaling + self.elas_offset

        out.color = self.default_color
        if self.random_color:
            out.color = random_bright_color()

        out.scale = self.default_scale
        if self.random_scale:
            out.scale = np.random.rand() * self.scale_scaling + self.scale_offset
            # TODO: scale the vertices of the OBJ files on the fly here

        urdf_str = fill_urdf(
            out.obj,
            obj_scale=out.scale,
            color=out.color,
            mass=out.mass,
            rolling_fric=out.fric,
            lateral_fric=out.fric,
            spinnin_fric=out.fric,
            restitution=out.elas,
        )
        out.urdf = write_tmp_urdf(urdf_str)

        return out

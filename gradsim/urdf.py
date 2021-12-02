import os
import tempfile
from datetime import datetime

from gradsim.utils.misc import random_bright_color, random_string

URDF_TEMPLATE = """
<?xml version="1.0" encoding="utf-8"?>
<robot name="{name}">
    <link name="base_link">
        <visual>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{obj_file_visual}" scale="{obj_scale}"/>
            </geometry>
            <material name="color">
                <color rgba="{color}"/>
            </material>
        </visual>

        <collision>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{obj_file_collis}" scale="{obj_scale}"/>
            </geometry>
        </collision>

        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value="{mass}"/>
            <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
        </inertial>

        <contact>
            {contact_params}
        </contact>
    </link>
</robot>
"""

URDF_ROLLING_FRIC = "<rolling_friction value='{}'/>"
URDF_LATERAL_FRIC = "<lateral_friction value='{}'/>"
URDF_SPINNIN_FRIC = "<spinning_friction value='{}'/>"
URDF_RESTITUTION = "<restitution value='{}'/>"
URDF_STIFFNESS = "<stiffness value='{}'/>"
URDF_DAMPING = "<damping value='{}'/>"

URDF_CONTACT_SETTINGS = {
    "rolling_friction": URDF_ROLLING_FRIC,
    "lateral_friction": URDF_LATERAL_FRIC,
    "spinnin_friction": URDF_SPINNIN_FRIC,
    "restitution": URDF_RESTITUTION,
    "stiffness": URDF_STIFFNESS,
    "damping": URDF_DAMPING,
}


class MomentsOfInertia(object):
    def __init__(self, mass=None, ixx=None, iyy=None, izz=None):
        super().__init__()

        self.mass = mass
        # TODO calculate MoO on the fly for different body types, see xacro file
        self.ixx = ixx
        self.iyy = iyy
        self.izz = izz

        # FIXME: shitty intermediate solution if inertia aren't precalculated
        if ixx is None:
            self.ixx = 1
        if iyy is None:
            self.iyy = 1
        if izz is None:
            self.izz = 1


def fill_urdf(
    obj_visual,
    obj_scale=1,
    obj_collis=None,
    color=None,
    mass=1,
    inertia=None,
    rolling_fric=None,  # this could've been done with the URDF_CONTACT_SETTINGS dictionary and kwargs
    lateral_fric=None,  # but fuck that. I like having readable function parameters
    spinnin_fric=None,
    restitution=None,
    stiffness=None,
    damping=None,
):

    if obj_collis is None:
        obj_collis = obj_visual

        # TODO run pybullet V-HACD

        ###
        # import pybullet as p
        # import pybullet_data as pd
        # import os
        #
        # p.connect(p.DIRECT)
        # name_in = os.path.join(pd.getDataPath(), "duck.obj")
        # name_out = "duck_vhacd2.obj"
        # name_log = "log.txt"
        # p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=50000 ) # and then store this

    if color is None:
        r, g, b = random_bright_color()
        color = f"{r} {g} {b}"

    if inertia is None:
        moo = MomentsOfInertia(mass)
    else:
        moo = inertia

    contact_params = []
    if rolling_fric is not None:
        contact_params.append(URDF_ROLLING_FRIC.format(rolling_fric))
    if lateral_fric is not None:
        contact_params.append(URDF_LATERAL_FRIC.format(lateral_fric))
    if spinnin_fric is not None:
        contact_params.append(URDF_SPINNIN_FRIC.format(spinnin_fric))
    if restitution is not None:
        contact_params.append(URDF_RESTITUTION.format(restitution))
    if stiffness is not None:
        contact_params.append(URDF_STIFFNESS.format(stiffness))
    if damping is not None:
        contact_params.append(URDF_DAMPING.format(damping))

    contact_params = "\n".join(contact_params)

    urdf = URDF_TEMPLATE.format(
        obj_file_visual=obj_visual,
        obj_file_collis=obj_collis,
        obj_scale=obj_scale,
        color=color,
        name=random_string(),
        mass=mass,
        ixx=moo.ixx,
        iyy=moo.iyy,
        izz=moo.izz,
        contact_params=contact_params,
    )

    return urdf


# we can reuse this
tmp = tempfile.gettempdir()
urdf_dir = os.path.join(tmp, "gradsim-urdf")


def write_tmp_urdf(urdf):
    now = datetime.now()
    millisec = str(int(round(float(now.strftime("%f")) / 1000)))
    timestamp = now.strftime("%Y%m%d-%H%M%S-") + millisec
    if not os.path.isdir(urdf_dir):
        os.mkdir(urdf_dir)
    path = os.path.join(urdf_dir, f"{timestamp}.urdf")
    with open(path, "w") as f:
        f.write(urdf)

    return path

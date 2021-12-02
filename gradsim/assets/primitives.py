import os
from enum import Enum

from gradsim.assets import ASSET_DIR


class Primitive(Enum):
    BOX = 1
    CAPSULE = 2
    CONE = 3
    CYLINDER = 4
    SPHERE = 5


PRIMITIVE_DICT = {
    Primitive.BOX: "box",
    Primitive.CAPSULE: "capsule",
    Primitive.CONE: "cone",
    Primitive.CYLINDER: "cylinder",
    Primitive.SPHERE: "sphere",
}


INT_TO_PRIMITIVE = {
    1: Primitive.BOX,
    2: Primitive.CAPSULE,
    3: Primitive.CONE,
    4: Primitive.CYLINDER,
    5: Primitive.SPHERE,
}


def get_path(obj):
    return os.path.join(ASSET_DIR, f"{obj}.obj")


def get_primitive_obj(p):
    assert isinstance(p, Primitive)
    return get_path(PRIMITIVE_DICT[p])

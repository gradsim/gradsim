"""
Define simulation defaults.
"""

from collections import namedtuple

from gradsim.assets.primitives import Primitive

Defaults = namedtuple(
    "Defaults",
    [
        "FRICTION_COEFFICIENT",
        "RESTITUTION",
        "EPSILON",
        "TOL",
        "FPS",
        "DT",
        "SHAPE",
        "MASS",
        "FRICTION",
        "COLOR",
        "SCALE",
    ],
)

# Coefficient of friction.
Defaults.FRICTION_COEFFICIENT = 0.9
# Coefficient of restitution.
Defaults.RESTITUTION = 0.5
# Contact detection threshold.
Defaults.EPSILON = 0.1
# Penetration tolerance parameter.
Defaults.TOL = 1e-6
# Threshold relative velocity below which collision will not be corrected
# (i.e., will be treated as a resting contact).
Defaults.VREL_THRESH = 0.01
# Frames per second.
Defaults.FPS = 30
# Default time step sizes.
Defaults.DT = 1.0 / Defaults.FPS
# Default object mass if not random
Defaults.MASS = 0.15
# Default object shape
Defaults.SHAPE = Primitive.BOX
# I'm not gonna comment on what this next line does
Defaults.COLOR = (255, 0, 255)
# Default object scale
Defaults.SCALE = 1

import copy
import logging
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGE_NAME = "gradsim"
VERSION = "0.0.2"
DESCRIPTION = "gradsim: A differentiable 2D physics simulator"
URL = "<url.to.go.in.here>"
AUTHOR = "Krishna Murthy Jatavallabhula"
LICENSE = "(TBD)"
DOWNLOAD_URL = ""
LONG_DESCRIPTION = """
A differentiable 3D rigid-body simulator (physics and rendering engines).
"""
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX :: Linux",
    # TODO: Add Windows OS
    "Programming Language :: Python :: 3 :: Only",
    "License :: OSI Approved :: MIT",
    "Topic :: Software Development :: Libraries",
]

cwd = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger()
logging.basicConfig(format="%(levelname)s - %(message)s")

# Minimum required pytorch version (TODO: check if versions below this work).
TORCH_MINIMUM_VERSION = "1.3.0"

# Check that PyTorch version installed meets minimum requirements.
if torch.__version__ < TORCH_MINIMUM_VERSION:
    logger.warning(
        f"gradsim has beent tested with PyTorch >= {0}. Found version {1} instead.".format(
            TORCH_MINIMUM_VERSION, torch.__version__
        )
    )

has_cuda = True

if not torch.cuda.is_available():
    has_cuda = False
    # From: https://github.com/NVIDIA/apex/blob/b66ffc1d952d0b20d6706ada783ae5b23e4ee734/setup.py
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    logging.warning(
        "\nWarning: Torch did not find available GPUs on this system.\n"
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, Kaolin will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), and Turing (compute capability 7.5).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n'
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"


def build_deps():
    print("Building nv-usd...")
    os.system("./buildusd.sh")


def CustomCUDAExtension(*args, **kwargs):
    if not os.name == "nt":
        FLAGS = ["-Wno-deprecated-declarations"]
        kwargs = copy.deepcopy(kwargs)
        if "extra_compile_args" in kwargs:
            kwargs["extra_compile_args"] += FLAGS
        else:
            kwargs["extra_compile_args"] = FLAGS

    return CUDAExtension(*args, **kwargs)


class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        # ninja is interfering with compiling separate extensions in parallel
        kwargs["use_ninja"] = False
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        if not os.name == "nt":
            FLAG_BLACKLIST = ["-Wstrict-prototypes"]
            FLAGS = ["-Wno-deprecated-declarations"]
            self.compiler.compiler_so = [
                x for x in self.compiler.compiler_so if x not in FLAG_BLACKLIST
            ] + FLAGS  # Covers non-cuda

        super().build_extensions()


def get_extensions():
    cuda_extensions = [
        CustomCUDAExtension(
            "gradsim.renderutils.cuda.load_textures",
            [
                "gradsim/renderutils/cuda/load_textures_cuda.cpp",
                "gradsim/renderutils/cuda/load_textures_cuda_kernel.cu",
            ],
        ),
        CustomCUDAExtension(
            "gradsim.renderutils.cuda.soft_rasterize_cuda",
            [
                "gradsim/renderutils/cuda/soft_rasterize_cuda.cpp",
                "gradsim/renderutils/cuda/soft_rasterize_cuda_kernel.cu",
            ],
        ),
        CustomCUDAExtension(
            "gradsim.renderutils.cuda.voxelization_cuda",
            [
                "gradsim/renderutils/cuda/voxelization_cuda.cpp",
                "gradsim/renderutils/cuda/voxelization_cuda_kernel.cu",
            ],
        ),
        CustomCUDAExtension(
            "gradsim.renderutils.dibr.cuda.rasterizer",
            [
                "gradsim/renderutils/dibr/cuda/rasterizer.cpp",
                "gradsim/renderutils/dibr/cuda/rasterizer_cuda.cu",
                "gradsim/renderutils/dibr/cuda/rasterizer_cuda_back.cu",
            ],
        ),
    ]
    return cuda_extensions


def get_requirements():
    return [
        "torch",
        "black",
        "flake8",
        "h5py",
        "imageio",
        "isort",
        "matplotlib",
        "numpy",
        "Pillow",
        "py3ode",
        "pygame",
        "pytest>=4.6",
        "pytest-cov>=2.7",
        "pyyaml",
        "sphinx==2.2.0",  # pinned to resolve issue with docutils 0.16b0.dev
        "torchdiffeq",
        "tqdm",
    ]


if __name__ == "__main__":
    kwargs = {}
    if has_cuda:
        kwargs_cuda = {
            "ext_modules": get_extensions(),
            "cmdclass": {"build_ext": CustomBuildExtension},
        }
        kwargs = {**kwargs, **kwargs_cuda}
    else:
        logging.warning(
            "Building without CUDA-based extensions.\n"
            "This means the SoftRas, the texture loader, and the voxelization won't work."
        )

    build_deps()  # Build USD bindings

    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        licence=LICENSE,
        python_requires=">3.6",
        # Package info
        packages=find_packages(exclude=("docs", "test", "examples")),
        install_requires=get_requirements(),
        zip_safe=True,
        classifiers=CLASSIFIERS,
        **kwargs,
    )

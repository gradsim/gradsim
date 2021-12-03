# gradsim

> gradSim: Differentiable simulation for system identification and visuomotor control

<p align="center">
	<img src="assets/walker.gif" />
</p>

**gradSim** is a unified differentiable rendering and multiphysics framework that allows solving a range of control and parameter estimation tasks (rigid bodies, deformable solids, and cloth) directly from images/video. Our unified computation graph — spanning from the dynamics and through the rendering process — enables learning in challenging visuomotor control tasks, without relying on state-based (3D) supervision, while obtaining performance competitive to or better than techniques that rely on precise 3D labels.


## Building the package


#### Step 0: Create a conda environment

We recommend creating a [`conda`](https://anaconda.org/) environment (or alternatively, a `virtualenv` environment) before proceeding. Recommended python versions are 3.6 or higher. Tested versions include python 3.6 and 3.7.
```
conda create -n gradsim python=3.7
conda activate gradsim
```

#### Step 1: Install `pytorch`

In a `conda` or `virtualenv` environment, first install your favourite version of `PyTorch` (versions `>=1.3.0` recommended; untested with lower) by following instructions from [pytorch.org](https://pytorch.org/)


#### Step 2: Install `kaolin`

Our code relies on NVIDIA [`kaolin`](https://github.com/NVIDIAGameWorks/kaolin) functionality. Please install NVIDIA Kaolin by following instructions from [the official docs](https://kaolin.readthedocs.io/en/latest/notes/installation.html).


#### Step 3: Build the `gradsim` package

After activating the environment (and installing `PyTorch` and `kaolin`), run
```bash
python setup.py build develop
```

This will install the `gradsim` package, which you can now use! To smoketest and verify that the package has been installed, fire up your python interpreter and run the following quick commands.

```py
>>> import gradsim
>>> gradsim.__version__
```


#### Step 4a: (IMPORTANT) Setup PYTHONPATH

`gradsim` heavily relies on the `Usd` package from `pxr`, which is installed by default when you install `kaolin`. For most of the examples to run, you will need to configure your `PYTHONPATH` environment variable so that the `pxr` module (and its `Usd` submodule) are available to python.

First, navigate to the directory where you cloned the `kaolin` repository.
```
cd path/to/kaolin/root/directory
```

In this directory, execute the following commands:
```
export KAOLIN_HOME=$PWD
export PYTHONPATH=${KAOLIN_HOME}/'build/target-deps/nv_usd/release/lib/python'
export LD_LIBRARY_PATH=${KAOLIN_HOME}/'build/target-deps/nv_usd/release/lib/'
```

#### Step 4b (alternative)

> **Only do this if you have NOT followed step 4a. If you have already followed step 4a, skip this step**

Instead of step 4a, you may use the provided installer to install the `pxr` module (and its `Usd` submodule).
```
./buildusd.sh
```

IMPORTANT: After the above step, run the following command from the base directory of this repo (i.e., directory containing this `README` file)
```
source setenv.sh
```
This script will setup your `PYTHONPATH` and `LD_LIBRARY_PATH` environment variables.


#### Step 5: Install Ninja

The [Ninja build system](https://ninja-build.org/) must be installed prior to running many of our demos.
```
conda install -c conda-forge ninja
```



#### Known issues

On Mac systems, if Py3ODE is not installed, the build of that package might fail, in which case, please clone their github repo into a separate folder:

```bash
cd .. # assuming you are currently in the directory of this project
git clone https://github.com/filipeabperes/Py3ODE.git
cd Py3ODE
```

And compile and install their Python module,...

```bash
CC=/usr/bin/clang CFLAGS="-O -stdlib=libc++ \
-I/Library/Developer/CommandLineTools/usr/include/c++/v1/" \
python setup.py build

python setup.py install
```


## Differentiable simulation for visual system identification and visuomotor control

Our code release includes a number of examples. Here is a brief description of each example, and a sample launch command.
> NOTE: Each script takes in a number of commandline arguments. So please read through the code for a complete idea.

> NOTE: All of the following scripts are contained in the `examples` directory

#### hellogradsim.py

This example demonstrates a cube falling under the influence of gravity. Run this using the following command. Best run from inside the `examples` directory.

```bash
cd examples
python hellogradsim.py
```

This will save a rendered `gif` at `examples/cache/hellogradsim.gif`. The quality of this gif might be a tad poor if using the `cube.obj` model (from `examples/sampledata/cube.obj`) as it has just 12 triangles.

#### demo_forces.py

This examples demonstrates how to apply forces at arbitrary chosen points on a rigid body, as opposed to only the center of mass (such forces generate rotations, etc. which are of interest).

```bash
cd examples
python demo_forces.py
```

#### demo_mass_known_shape.py

This example demonstrates the usage of `gradsim` to recover the (unknown) mass of an object (represented as a triangle mesh), acted upon by an arbitrarily chosen force.

```bash
cd examples
python demo_mass_known_shape.py
```

For a list of available commandline arguments, run

```bash
python demo_mass_known_shape.py --help
```

#### demo_pendulum.py

This examples optimizes for the length of a simple pendulum from a video sequence.

```bash
python demo_pendulum.py
```

For a list of available commandline arguments, run

```bash
python demo_pendulum.py --help
```

#### demo_double_pendulum.py

This examples optimizes for the lengths of a double pendulum from a video sequence.

```bash
python demo_double_pendulum.py
```

For a list of available commandline arguments, run

```bash
python demo_double_pendulum.py --help
```

#### demo_bounce2d.py

A 2D bouncing ball demo. Assumes the ball is bounced with an initial horizontal velocity, and that gravity is the only external force acting on it.

```bash
python demo_bounce2d.py
```

For a list of available commandline arguments, run

```bash
python demo_bounce2d.py --help
```

#### hellodflex.py

A simple `dflex` smoketest. If `dflex` is being used for the first time, all of the kernels will be compiled.

```bash
python hellodflex.py
```

#### demo_fem.py

Optimizes the parameters of an FEM model (uses the NeoHookean hyperelasticity model). In particular, the demo shows how to optimize for the mass of each particle in the deformable mesh to math a rendered video sequence.

```bash
python demo_fem.py
```

#### demo_cloth.py

A cloth parameter optimization demo. Estimate the velocity of each particle in a cloth mesh, to match a target rendering (video sequence).

```bash
python demo_cloth.py
```

#### control_fem.py

A visuomotor control demo that trains a neural network to actuate the tetrahedrons of a `gear` mesh, in order for it to achieve a specified target pose.

```bash
python control_fem.py
```

#### control_cloth.py

A visuomotor control demo that optimizes the initial velocity of a cloth, in order for it to achieve a specified target pose.

```bash
python control_cloth.py
```

#### control_walker.py

A visuomotor control demo that optimizes the actuation phases of a walker, to make it walk towards a target pose (specified as an image).

```bash
python control_walker.py
```


#### exp01_*.py

Files used for running experiment 1 of the gradSim ICLR submission


#### exp*_plotlandscape.py

Scripts that plot loss landscapes found in the paper / appendix.


#### paperviz_*.py

A subset of scripts used to generate visualizations for the paper.


## Misc scripts/plotting resources

In the `plots` folder, we release all data and scripts used in the plots for the ICLR version of our paper.


## Citing gradSim

For attribution in academic contexts, please cite this work as

Citation

```
Jatavallabhula and Macklin et al., "gradSim: Differentiable simulation for system identification and visuomotor control", ICLR 2021.

```

BibTeX citation

```
@article{gradsim,
  title   = {gradSim: Differentiable simulation for system identification and visuomotor control},
  author  = {Krishna Murthy Jatavallabhula and Miles Macklin and Florian Golemo and Vikram Voleti and Linda Petrini and Martin Weiss and Breandan Considine and Jerome Parent-Levesque and Kevin Xie and Kenny Erleben and Liam Paull and Florian Shkurti and Derek Nowrouzezahrai and Sanja Fidler},
  journal = {International Conference on Learning Representations (ICLR)},
  year    = {2021},
  url     = {https://openreview.net/forum?id=c_E8kFWfhp0},
  pdf     = {https://openreview.net/pdf?id=c_E8kFWfhp0},
}
```

If you find this code useful, please also consider citing Kaolin and DFlex.

```
@misc{kaolin,
author = {Clement Fuji-Tsang and Masha Shugrina and Jean-Francois Lafleche and Charles Loop and Towaki Takikawa and Jiehan Wang and Wenzheng Chen and Sanja Fidler and Jason Gorski and Rev Lebaredian and Jianing Li and Michael Li and Krishna Murthy Jatavallabhula and Artem Rozantsev and Frank Shen and Edward Smith and Gavriel State and Tommy Xiang},
title = {Kaolin: A PyTorch Library for Accelerating 3D Deep Learning Research},
howpublished = {\url{https://github.com/NVIDIAGameWorks/kaolin}},
year = {2019}
}
```

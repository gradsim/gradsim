import argparse
import glob
import os
import random
import shutil
import sys
import time

import imageio
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import trange

from gradsim.bodies import RigidBody
from gradsim.forces import ConstantForce
from gradsim.generator import PrimitiveGenerator
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.simulator import Simulator
from gradsim.utils import meshutils
from gradsim.utils.defaults import Defaults
from gradsim.utils.h5 import HDF5Dataset, HDF5Maker
from gradsim.engines import EulerIntegrator, SemiImplicitEulerWithContacts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./h5",
        help="Directory where h5 files are to be saved: out_dir/shard_0001.hdf5, etc.",
    )
    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sampledata",
        help="Directory with primitive .obj files, or containing dirs of shapenet",
    )
    parser.add_argument("--data_type", type=str, default="primitive",
                        choices=["primitive", "shapenet", "obj"])
    parser.add_argument("--target_vertices", type=int, default=None)
    parser.add_argument(
        "--image_size", type=int, default=256, help="size of rendered image in pixels"
    )
    parser.add_argument("--gpu", type=int, default=0, help="ID of gpu to use")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed (for random number generators)",
    )
    # Simulation
    parser.add_argument(
        "--sim_duration", type=float, default=2, help="# of seconds of video to render"
    )
    parser.add_argument("--sim_sub_steps", type=int, default=1)
    parser.add_argument("--sim_fps", type=int, default=Defaults.FPS)
    parser.add_argument("--restitution", type=eval, default=False, choices=[True, False])
    parser.add_argument(
        "--n", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument("--record_trajectory", type=eval, default=False, choices=[True, False])
    parser.add_argument("--volume_thresh", type=float, default=0.5)
    # Body
    parser.add_argument(
        "--random_init_pos", type=eval, default=True, choices=[True, False]
    )
    parser.add_argument("--init_x_scaling", type=float, default=4)
    parser.add_argument("--init_x_offset", type=float, default=-2)
    parser.add_argument("--init_y_scaling", type=float, default=1.5)
    parser.add_argument("--init_y_offset", type=float, default=3.5)
    parser.add_argument(
        "--random_orientation", type=eval, default=True, choices=[True, False]
    )
    parser.add_argument(
        "--random_linear_velocity", type=eval, default=True, choices=[True, False]
    )
    parser.add_argument(
        "--random_angular_velocity", type=eval, default=True, choices=[True, False]
    )
    parser.add_argument("--mass_scaling", type=float, default=0.1)
    parser.add_argument("--mass_offset", type=float, default=0.2)
    # Gravity
    parser.add_argument("--gravity", type=eval, default=False, choices=[True, False])
    parser.add_argument("--gravity_magnitude", type=float, default=10.0, help="Gravity magnitude")
    # Impulse
    parser.add_argument(
        "--random_impulse_magnitude", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument(
        "--random_impulse_direction", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument("--impulse_magnitude", type=float, default=10.0, help="Impulse magnitude, in case not random")
    parser.add_argument("--impulse_direction", type=str, default="0,-1,0", help="Impulse direction, in case not random")
    # Camera
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="look_at",
        choices=["projection", "look", "look_at"],
    )
    parser.add_argument("--camera_distance", type=float, default=12.0)
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--azimuth", type=float, default=0.0)
    # HDF5
    parser.add_argument(
        "--compression", type=str, default="gzip", choices=["None", "gzip", "lzf"]
    )
    parser.add_argument("--num_per_shard", type=int, default=10000)
    parser.add_argument("--force", type=eval, default=False, choices=[True, False])
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (i.e., quickly generate a few samples to get a sense)",
    )
    parser.add_argument("--debugdir", type=str, default=".")
    parser.add_argument("--debug_num", type=int, default=10)
    # Parse
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir)
    args.command = "python " + " ".join(sys.argv)
    args.sim_dtime = (1 / args.sim_fps) / args.sim_sub_steps
    args.sim_steps = int(args.sim_duration / args.sim_dtime)
    args.n_frames = int(args.sim_duration * args.sim_fps)
    args.impulse_direction = np.array(list(map(float, args.impulse_direction.split(','))))
    if args.compression == "None":
        args.compression = None
    return args


def define_props(
    random_init_pos,
    random_orientation,
    random_linear_velocity,
    random_angular_velocity,
    random_force_magnitude,
    random_force_direction,
    default_force_magnitude=10.0,
    default_force_direction=np.array([0.0, -1.0, 0.0]),
    init_x_scaling=4,
    init_x_offset=-2,
    init_y_scaling=1.0,
    init_y_offset=2.0,
):
    # Init position
    init_pos = (
        [
            (torch.rand(1).item() * init_x_scaling) + init_x_offset,
            (torch.rand(1).item() * init_y_scaling) + init_y_offset,
            0,
        ]
        if random_init_pos
        else [0.0, 0.0, 0.0]
    )
    # Orientation
    orientation = (
        [
            1 + (torch.rand(1).item() * 2 - 1),
            (torch.rand(1).item() * 2 - 1),
            (torch.rand(1).item() * 2 - 1),
            (torch.rand(1).item() * 2 - 1),
        ]
        if random_orientation
        else [1.0, 0.0, 0.0, 0.0]
    )
    # Velocity
    linear_velocity = (
        [(torch.rand(1).item() * 2 - 1) * 10, (torch.rand(1).item() * 2 - 1) * 2, 0]
        if random_linear_velocity
        else [0.0, 0.0, 0.0]
    )
    angular_velocity = (
        [(torch.rand(1).item() * 2 - 1) * 10, (torch.rand(1).item() * 2 - 1) * 2, 0]
        if random_angular_velocity
        else [0.0, 0.0, 0.0]
    )
    # Force
    force_magnitude = (
        max(0.0, default_force_magnitude + torch.randn(1).item()) if random_force_magnitude else default_force_magnitude
    )
    if random_force_direction:
        force_direction = np.random.rand(3) * 2 - 1
        force_direction /= np.linalg.norm(force_direction)
    else:
        force_direction = default_force_direction
    return (
        init_pos,
        orientation,
        linear_velocity,
        angular_velocity,
        force_magnitude,
        force_direction,
    )


def make_rigid_body(sample, init_pos, orientation, linear_velocity, angular_velocity,
                    device, target_vertices=None):
    # Load a triangle mesh obj file
    if target_vertices is not None:
        tmp_obj_file = '/tmp/obj.obj'
        from blender_process import Process
        p = Process(sample.obj, target_vertices, tmp_obj_file)
        mesh = TriangleMesh.from_obj(tmp_obj_file)
    else:
        mesh = TriangleMesh.from_obj(sample.obj)
    vertices = (
        meshutils.normalize_vertices(mesh.vertices)
        .unsqueeze(0)
        .to(device)
    )
    faces = mesh.faces.unsqueeze(0).to(device)
    textures = torch.cat(
        (
            sample.color[0]
            / 255
            * torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
            sample.color[1]
            / 255
            * torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
            sample.color[2]
            / 255
            * torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),
        ),
        dim=-1,
    )
    # (Uniform) Masses
    masses = (sample.mass / vertices.shape[-2]) * torch.nn.Parameter(
        torch.ones(vertices.shape[-2], dtype=vertices.dtype, device=device),
        requires_grad=True,
    )
    # Body
    body = RigidBody(
        vertices[0],
        position=torch.tensor(init_pos, dtype=torch.float32, device=device),
        masses=masses,
        orientation=torch.tensor(orientation).type(torch.float32).to(device),
        friction_coefficient=sample.fric,
        # linear_velocity=torch.tensor(linear_velocity).type(torch.float32).to(device)
        # angular_veloctiy=torch.tensor(angular_velocity).type(torch.float32).to(device)
    )
    return body, vertices, faces, textures


def render_samples(sim_steps, sim_dtime, render_every, restitution, renderer, body, faces, textures,
                   record_trajectory=False):
    # Initialize the simulator with the body at the origin.
    sim = Simulator(
            [body],
            dtime=sim_dtime,
            engine=SemiImplicitEulerWithContacts() if restitution else EulerIntegrator()
            )
    # Run the simulation
    sequence = []
    if record_trajectory:
        poss, ornts, linvels, angvels = [], [], [], []
        def record(body):
            poss.append(body.position.detach().cpu())
            ornts.append(body.orientation.detach().cpu())
            linvels.append(body.linear_velocity.detach().cpu())
            angvels.append(body.angular_velocity.detach().cpu())
    for t in trange(sim_steps, leave=False):
        sim.step()
        if t % render_every == 0:
            if record_trajectory:
                record(body)
            rgba = renderer.forward(body.get_world_vertices().unsqueeze(0), faces, textures)
            img = (rgba[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            sequence.append(img)
    if record_trajectory:
        f = lambda v: torch.stack(v).numpy()
        return sequence, f(poss), f(ornts), f(linvels), f(angvels)
    else:
        return sequence


def copy_scripts(src, dst):
    print("Copying scripts in", src, "to", dst)
    os.makedirs(dst, exist_ok=True)
    for file in (
        glob.glob(os.path.join(src, "*.sh"))
        + glob.glob(os.path.join(src, "*.py"))
        + glob.glob(os.path.join(src, "*.so"))
        + glob.glob(os.path.join(src, "*.cu"))
        + glob.glob(os.path.join(src, "*.cpp"))
        + glob.glob(os.path.join(src, "*.obj"))
    ):
        shutil.copy(file, dst)
    for d in glob.glob(os.path.join(src, "*/")):
        if (
            "__" not in os.path.basename(os.path.dirname(d))
            and "." not in os.path.basename(os.path.dirname(d))[0]
            and "ipynb" not in os.path.basename(os.path.dirname(d))
            and os.path.basename(os.path.dirname(d)) != "data"
            and os.path.basename(os.path.dirname(d)) != "experiments"
        ):
            if os.path.abspath(d) in os.path.abspath(dst):
                continue
            print("Copying", d)
            # shutil.copytree(d, os.path.join(dst, d))
            new_dir = os.path.join(dst, os.path.basename(os.path.normpath(d)))
            copy_scripts(d, new_dir)


if __name__ == "__main__":

    args = get_args()

    # Device to store tensors on (MUST be CUDA-capable, for renderer to work).
    device = f"cuda:{args.gpu}"

    # Seed random number generators
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize h5
    h5_maker = (
        HDF5Maker(
            args.out_dir,
            args.num_per_shard,
            compression=args.compression,
            force=args.force,
        )
        if not args.debug
        else None
    )

    # Save args
    if not args.debug:
        # copy_scripts(
        #     os.path.abspath(
        #         os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gradsim")
        #     ),
        #     os.path.join(args.out_dir, "gradsim"),
        # )
        shutil.copy(os.path.abspath(__file__), args.out_dir)
        os.makedirs(os.path.join(args.out_dir, "obj"), exist_ok=True)
        for obj in glob.glob(os.path.join(args.data_dir, "*.obj")):
            shutil.copy(obj, os.path.join(args.out_dir, "obj"))
        with open(os.path.join(args.out_dir, "args.yaml"), "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    # Initialize the renderer.
    renderer = SoftRenderer(
        image_size=args.image_size, camera_mode=args.camera_mode, device=device
    )
    renderer.set_eye_from_angles(args.camera_distance, args.elevation, args.azimuth)

    # Initialize sampler
    prim_gen = PrimitiveGenerator(
        mass_scaling=args.mass_scaling, mass_offset=args.mass_offset,
        data_type=args.data_type, data_dir=args.data_dir
    )

    objects = sorted(glob.glob(os.path.join(args.data_dir, "*.obj")))

    pos, ornt, linvel, angvel = None, None, None, None

    try:
        for i in trange(args.n):

            volume = 0.0
            while volume < args.volume_thresh:

                # Sample
                # def s():
                #     sample = prim_gen.sample()
                #     if sample.shape._value_ == 1:
                #         print("returning")
                #         return sample
                #     else:
                #         return s()
                # sample = s()
                sample = prim_gen.sample()

                # Define props
                (
                    init_pos,
                    orientation,
                    linear_velocity,
                    angular_velocity,
                    impulse_magnitude,
                    impulse_direction,
                ) = define_props(
                    args.random_init_pos,
                    args.random_orientation,
                    args.random_linear_velocity,
                    args.random_angular_velocity,
                    args.random_impulse_magnitude,
                    args.random_impulse_direction,
                    args.impulse_magnitude,
                    args.impulse_direction,
                    args.init_x_scaling,
                    args.init_x_offset,
                    args.init_y_scaling,
                    args.init_y_offset,
                )

                # Body
                body, vertices, faces, textures = make_rigid_body(
                    sample, init_pos, orientation, linear_velocity, angular_velocity,
                    device, args.target_vertices
                )

                volume = (vertices[0].max(0)[0] - vertices[0].min(0)[0]).norm().item()
                if args.debug:
                    print("Volume:", volume)

            # Impulse application points
            # application_points = [0, 1]
            # application_points = [
            #     int(v)
            #     for v in torch.randint(
            #         vertices.shape[1], (max(1, torch.randint(vertices.shape[1], (1,))),)
            #     )
            # ]
            inds = vertices.argmin(1)
            application_points = list(inds.view(-1).detach().cpu().numpy())
            application_points = [application_points[1]]

            # Add an impulse
            if impulse_magnitude > 0:
                impulse = ConstantForce(
                    magnitude=impulse_magnitude,
                    direction=torch.from_numpy(impulse_direction),
                    starttime=0.0,
                    endtime=0.1,
                    device=device,
                )
                body.add_external_force(impulse, application_points=application_points)

            # Add gravity
            if args.gravity:
                gravity = ConstantForce(
                    magnitude=args.gravity_magnitude / len(vertices[0]),
                    direction=torch.tensor([0, -1, 0]),
                    device=device,
                )
                body.add_external_force(gravity)

            # Make sequence 
            sequence = render_samples(
                sim_steps=args.sim_steps,
                sim_dtime=args.sim_dtime,
                render_every=args.sim_sub_steps,
                restitution=args.restitution,
                renderer=renderer,
                body=body,
                faces=faces,
                textures=textures,
                record_trajectory=args.record_trajectory,
            )
            if args.record_trajectory:
                sequence, init_pos, orientation, linear_velocity, angular_velocity = sequence

            if args.debug:
                print(sample.__dict__, init_pos[0] if args.record_trajectory else init_pos)
                imageio.mimwrite(os.path.join(args.debugdir, f"{i:02d}.gif"), sequence)
                if i > args.debug_num:
                    sys.exit()

            else:
                # Save the sequence
                sequence = np.array(sequence).astype(np.uint8)
                # Shape
                if args.data_type == "primitive":
                    shape = sample.shape._value_
                else:
                    shape = sample.shape
                # Add to dataset
                h5_maker.add_data(
                    sequence=sequence,
                    shape=shape,
                    position=init_pos,
                    orientation=orientation,
                    mass=sample.mass,
                    fric=sample.fric,
                    elas=sample.elas,
                    color=sample.color,
                    scale=sample.scale,
                    force_application_points=application_points,
                    force_magnitude=impulse_magnitude,
                    force_direction=impulse_direction,
                    linear_velocity=linear_velocity,
                    angular_velocity=angular_velocity,
                    random_force_magnitude=args.random_impulse_magnitude,
                    random_force_direction=args.random_impulse_direction,
                    random_linear_velocity=args.random_linear_velocity,
                    random_angular_velocity=args.random_angular_velocity,
                    objects=objects,
                    camera_distance=args.camera_distance,
                    elevation=args.elevation,
                    azimuth=args.azimuth,
                )

    except KeyboardInterrupt:
        pass

    if not args.debug:
        h5_maker.close()

    # Load 10 samples of only sequences
    h5_ds = HDF5Dataset(args.out_dir, read_only_seqs=True)
    h5_dl = DataLoader(
        h5_ds, batch_size=10, shuffle=True, collate_fn=h5_ds.HDF5_collate_fn
    )
    s = time.time()
    seqs = next(iter(h5_dl))
    e = time.time()
    print(e - s)

# Load 10 samples
h5_ds = HDF5Dataset(args.out_dir)
h5_dl = DataLoader(
    h5_ds, batch_size=10, shuffle=True, collate_fn=h5_ds.HDF5_collate_fn
)
s = time.time()
out = next(iter(h5_dl))
e = time.time()
print(e - s)
(
    seqs,
    shape,
    position,
    orientation,
    mass,
    fric,
    elas,
    color,
    scale,
    force_application_points,
    force_magnitude,
    force_direction,
    linear_velocity,
    angular_velocity,
) = out

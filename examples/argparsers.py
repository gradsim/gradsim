import argparse
import os


def get_dflex_base_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expid",
        type=str,
        default="default",
        help="Unique string identifier for this experiment.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.join("cache", "control"),
        help="Directory to store experiment logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (for repeatability)."
    )
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=3.0,
        help="Duration of the simulation episode.",
    )
    parser.add_argument(
        "--physics-engine-rate",
        type=int,
        default=60,
        help="Number of physics engine `steps` per 1 second of simulator time.",
    )
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=32,
        help="Number of sub-steps to integrate, per 1 `step` of the simulation.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training iterations."
    )
    parser.add_argument("--lr", type=float, default=10, help="Learning rate.")
    parser.add_argument("--log", action="store_true", help="Log experiment data.")

    return parser

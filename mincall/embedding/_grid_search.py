#!/usr/bin/env python3

from tqdm import tqdm
import argparse
import subprocess
import os
from sklearn.model_selection import ParameterGrid

hyper_params = {
    'window': {2, 3},
    'stride': {2, 5, 10},
    'receptive_field': {10, 15, 20},
    'embedding_size': {5, 10},
}

template =\
"""
version: "v0.1"
embed:
  window: {window}
  stride: {stride}
  receptive_field: {receptive_field}
  embedding_size: {embedding_size}
  train_steps: 10000
  batch_size: 64
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="/tmp/kk")
    parser.add_argument(
        "--root_data",
        default="/home/lpp/Desktop/resquigg",
        type=os.path.abspath
    )
    parser.add_argument("--rel_data", default=".")
    parser.add_argument("--image", default="nmiculinic/mincall:latest-py3")
    args = parser.parse_args()

    if os.path.isabs(args.rel_data):
        raise ValueError("rel_data must be abs path!")

    for params in tqdm(ParameterGrid(hyper_params)):
        fname = "__".join(f"{k}_{params[k]}" for k in sorted(params.keys()))
        folder = os.path.normpath(
            os.path.abspath(os.path.join(args.log_dir, fname))
        )
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "config.yml"), "w") as f:
            print(template.format(**params), file=f)
        cmd = [
            "docker",
            "run",
            "--rm",
            # f"-u={os.getuid()}:{os.getgid()}",
            "-v",
            f"{os.path.normpath(args.root_data)}:/data",
            "-v",
            f"{folder}:/logs",
            "-w",
            "/logs",
            args.image,
            "--config",
            "/logs/config.yml",
            "-f",
            os.path.normpath(os.path.join("/data", args.rel_data)),
        ]
        print(cmd)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

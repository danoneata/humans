import json
import os
import pdb

import numpy as np

import pandas as pd

from typing import Any, List, Optional, Tuple

import click

from tqdm import tqdm

from constants import LANDMARKS_INDICES, NUM_LANDMARKS
from data import Obama
from utils import make_folder


ESPNET_EGS_PATH = os.path.expanduser("~/src/espnet/egs2/obama")
CURRENT_DIR = os.path.expanduser("~/work/human")

DATASET = Obama()

LIPS_α = LANDMARKS_INDICES["lips"][0]
LIPS_β = LANDMARKS_INDICES["teeth"][1]
LIPS_IDXS = slice(LIPS_α, LIPS_β)
NUM_LIPS_IDXS = LIPS_β - LIPS_α


def get_face_landmarks_npy_path(landmarks_type, key):
    return os.path.join(
        "output", "obama", "face-landmarks-npy-" + landmarks_type, key + ".npy"
    )


def head(xs: List[Any]) -> np.ndarray:
    try:
        return np.stack(xs[0])
    except IndexError:
        return np.zeros((NUM_LANDMARKS, 2)) + np.nan


def interpolate_missing(xs):
    """Linearly interpolates missing landmarks."""
    xs = xs.reshape(-1, NUM_LIPS_IDXS * 2)
    df = pd.DataFrame(xs)
    df = df.interpolate(limit_direction="both")
    ys = df.to_numpy()
    ys = ys.reshape(-1, NUM_LIPS_IDXS, 2)
    return ys


def prepare_landmarks_npy(landmarks_type, key):
    path_i = DATASET.get_face_landmarks_path(key)
    path_o = get_face_landmarks_npy_path(landmarks_type, key)
    if os.path.exists(path_o):
        return key, path_o
    else:
        with open(path_i, "r") as f:
            landmarks = json.load(f)  # type: List[List[List[Tuple[int]]]]
        landmarks1 = np.stack(list(map(head, landmarks)))  # Landmarks of first face
        landmarks2 = landmarks1[:, LIPS_IDXS]  # Extract lips
        landmarks3 = interpolate_missing(landmarks2)
        make_folder(path_o)
        np.save(path_o, landmarks3)
        return key, path_o


@click.command()
@click.option("-s", "--split")
@click.option("-l", "--landmark-type", "landmark_type")
def main(split, landmark_type):
    keys = DATASET.load_filelist(split)
    data_scp = [prepare_landmarks_npy(landmark_type, key) for key in tqdm(keys)]
    path_scp = os.path.join(ESPNET_EGS_PATH, "data", "test", "lips.scp")
    make_folder(path_scp)
    with open(path_scp, "w") as f:
        for key, path in data_scp:
            f.write(key + " " + os.path.join(CURRENT_DIR, path) + "\n")


if __name__ == "__main__":
    main()

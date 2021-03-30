import json
import os
import pdb
import pickle

import numpy as np

import pandas as pd

from typing import Any, List, Optional, Tuple

import click

from tqdm import tqdm

from constants import LEN_LIPS, LIPS_INDICES, NUM_LANDMARKS
from data import Obama
from face_normalization import get_face_normalizer
from utils import make_folder


ESPNET_EGS_PATH = os.path.expanduser("~/src/espnet/egs2/obama")
CURRENT_DIR = os.path.expanduser("~/work/human")

DATASET = Obama()
LIPS_SLICE = slice(*LIPS_INDICES)


def get_face_landmarks_npy_path(landmarks_type, key, use_pca=False):
    suffix = "-pca" if use_pca else ""
    return os.path.join(
        "output", "obama", "face-landmarks-npy-" + landmarks_type + suffix, key + ".npy"
    )


def head(xs: List[Any]) -> np.ndarray:
    try:
        return np.stack(xs[0])
    except IndexError:
        return np.zeros((NUM_LANDMARKS, 2)) + np.nan


def interpolate_missing(xs):
    """Linearly interpolates missing landmarks."""
    if not np.any(np.isnan(xs)):
        return xs
    num_frames, num_landmarks, two = xs.shape
    xs = xs.reshape(num_frames, num_landmarks * two)
    df = pd.DataFrame(xs)
    df = df.interpolate(limit_direction="both")
    ys = df.to_numpy()
    ys = ys.reshape(num_frames, num_landmarks, two)
    return ys


def normalize(landmarks):
    landmarks_norm = np.zeros(landmarks.shape)
    for i, landmarks1 in enumerate(landmarks):
        normalizer = get_face_normalizer(landmarks1)
        landmarks_norm[i] = normalizer.forward(landmarks1)
    return landmarks_norm


def prepare_landmarks_npy(landmarks_type, key, overwrite=False, pca=None):
    use_pca = pca is not None

    path_i = DATASET.get_face_landmarks_path(key)
    path_o = get_face_landmarks_npy_path(landmarks_type, key, use_pca)

    if not overwrite and os.path.exists(path_o):
        return key, path_o

    with open(path_i, "r") as f:
        landmarks = json.load(f)  # type: List[List[List[Tuple[int]]]]

    landmarks1 = np.stack(list(map(head, landmarks)))  # Landmarks of first face
    landmarks1 = interpolate_missing(landmarks1)
    landmarks1 = normalize(landmarks1)
    landmarks1 = landmarks1[:, LIPS_SLICE]  # Extract lips
    landmarks1 = landmarks1.reshape(-1, LEN_LIPS * 2)


    make_folder(path_o)
    np.save(path_o, landmarks1)

    return key, path_o


def write_scp(path_scp, data_scp):
    make_folder(path_scp)
    with open(path_scp, "w") as f:
        for key, path in data_scp:
            f.write(key + " " + os.path.join(CURRENT_DIR, path) + "\n")


@click.command()
@click.option("-s", "--split")
@click.option("-l", "--landmarks-type", "landmarks_type")
@click.option("-w", "--overwrite", is_flag=True)
def main(split, landmarks_type, overwrite=False):
    keys = DATASET.load_filelist(split)
    data_scp = [
        prepare_landmarks_npy(landmarks_type, key, overwrite)
        for key in tqdm(keys)
    ]
    path_scp = os.path.join(ESPNET_EGS_PATH, "data", split, "lips.scp")
    write_scp(path_scp, data_scp)


if __name__ == "__main__":
    main()

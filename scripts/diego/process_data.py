import json
import os
import pdb
import subprocess

from collections import namedtuple

from typing import List

import numpy as np

import click

from constants import LEN_LIPS, LIPS_INDICES, NUM_LANDMARKS, Landmark
from data import DATASETS, Dataset
from face_normalization import LIPS_SLICE, get_first_face, normalize_landmarks
from scripts.obama.pca_landmarks import load_pca
from scripts.obama.prepare_landmarks_espnet import write_scp
from utils import make_folder


DATASET = DATASETS["diego-360p"]()


def get_face_landmarks_npy_path(
    dataset, landmarks_type: Landmark, key, use_pca: bool = False
):
    suffix = "-pca" if use_pca else ""
    return os.path.join(
        "output",
        dataset.name,
        "face-landmarks-npy-" + landmarks_type + suffix + "-reconstructed",
        dataset.key_to_str(key) + ".npy",
    )


def resize_video():
    if DATASET.video_res == "360p":
        size = "640x360"
    else:
        assert False
    for key in DATASET.load_filelist():
        src = DATASET.get_video_orig_path(key)
        dst = DATASET.get_video_path(key)
        if os.path.exists(dst):
            continue
        # fmt: off
        subprocess.run(
            [
                "ffmpeg",
                "-i", src,
                "-s", size,
                "-c:a", "copy",
                dst,
            ]
        )
        # fmt: on


def prepare_landmarks_npy(
    dataset: Dataset, landmarks_type: Landmark, key, overwrite=False, pca=None
):
    use_pca = pca is not None

    path_i = dataset.get_face_landmarks_path(key, landmarks_type)
    path_o = get_face_landmarks_npy_path(dataset, landmarks_type, key, use_pca)

    key_str = dataset.key_to_str(key)

    if not overwrite and os.path.exists(path_o):
        return key_str, path_o

    with open(path_i, "r") as f:
        landmarks = json.load(f)  # type: List[List[List[Tuple[int]]]]

    landmarks1 = np.stack(list(map(get_first_face, landmarks)))
    landmarks1 = normalize_landmarks(landmarks1)
    landmarks1 = landmarks1[:, LIPS_SLICE]  # Extract lips
    landmarks1 = landmarks1.reshape(-1, LEN_LIPS * 2)

    mask = np.all(~np.isnan(landmarks1), axis=1)
    landmarks1[mask] = pca.inverse_transform(pca.transform(landmarks1[mask]))
    landmarks1 = landmarks1.reshape(-1, LEN_LIPS, 2)

    make_folder(path_o)
    np.save(path_o, landmarks1)

    return key_str, path_o


def prepare_landmarks(pca=None):
    for key in DATASET.load_filelist():
        prepare_landmarks_npy(DATASET, "dlib", key, pca=pca, overwrite=True)


@click.command()
@click.option(
    "-t",
    "--todo",
    type=click.Choice(
        [
            "resize-video",
            "prepare-landmarks",
        ]
    ),
)
def main(todo, resolution=None):
    if todo == "resize-video":
        resize_video()
    elif todo == "prepare-landmarks":
        pca = load_pca()
        prepare_landmarks(pca=pca)
    else:
        assert False


if __name__ == "__main__":
    main()

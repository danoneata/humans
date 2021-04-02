import os
import pdb
import pickle

from typing import List, Tuple

import click

import numpy as np

from sklearn.decomposition import PCA  # type: ignore

from tqdm import tqdm

from constants import LEN_LIPS, LIPS_INDICES, SEED
from data import Obama
from scripts.obama.prepare_landmarks_espnet import (
    CURRENT_DIR,
    ESPNET_EGS_PATH,
    get_face_landmarks_npy_path,
    write_scp,
)
from utils import make_folder


np.random.seed(SEED)
NUM_LANDMARKS_PER_VIDEO = 128
DATASET = Obama()
LANDMARKS_TYPE = "dlib"


def get_pca_path(landmarks_type):
    return os.path.join("output", "obama", "pca", "pca-" + landmarks_type + ".pkl")


def load_pca(landmarks_type="dlib"):
    pca_path = get_pca_path(LANDMARKS_TYPE)
    with open(pca_path, "rb") as f:
        return pickle.load(f)


def fit():
    keys = DATASET.load_filelist("train")
    data = np.zeros((len(keys) * NUM_LANDMARKS_PER_VIDEO, LEN_LIPS, 2)) 

    for i, key in enumerate(keys):
        path = get_face_landmarks_npy_path("dlib", key)
        landmarks = np.load(path)

        idxs = np.random.choice(len(landmarks), size=NUM_LANDMARKS_PER_VIDEO)

        α = NUM_LANDMARKS_PER_VIDEO * i
        ω = NUM_LANDMARKS_PER_VIDEO * (i + 1)
        data[α: ω] = landmarks[idxs]

    data = data.reshape(-1, LEN_LIPS * 2)

    pca = PCA(n_components=8)
    pca.fit(data)

    path_pca = get_pca_path(landmarks_type="dlib")
    make_folder(path_pca)

    with open(path_pca, "wb") as f:
        pickle.dump(pca, f)


def transform(split):
    pca = load_pca(LANDMARKS_TYPE)

    data = []  # type: List[Tuple[str, str]]
    keys = DATASET.load_filelist(split)  # type: List[str]

    for key in tqdm(keys):
        path_i = get_face_landmarks_npy_path(LANDMARKS_TYPE, key)
        path_o = get_face_landmarks_npy_path(LANDMARKS_TYPE, key, use_pca=True)

        landmarks = np.load(path_i)
        landmarks = pca.transform(landmarks)

        np.save(path_o, landmarks)
        data.append((key, os.path.join(CURRENT_DIR, path_o)))

    path = os.path.join(ESPNET_EGS_PATH, "data", split, "lip.scp")
    write_scp(path, data)


@click.command()
@click.option("-t", "--todo", type=click.Choice(["fit", "transform"]))
@click.option("-s", "--split", type=click.Choice(["train", "valid", "test"]))
def main(todo, split):
    if todo == "fit":
        fit()
    elif todo == "transform":
        transform(split)
    else:
        assert False


if __name__ == "__main__":
    main()

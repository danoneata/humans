import json
import os
import pdb
import pickle

import numpy as np

import pandas as pd

from typing import Any, List, Optional, Tuple

import click

from tqdm import tqdm

from constants import LEN_LIPS, LIPS_INDICES, NUM_LANDMARKS, Landmark
from data import Dataset
from utils import make_folder


from constants import LANDMARKS_INDICES, LIPS_INDICES


LIPS_SLICE = slice(*LIPS_INDICES)


class Bijection:
    def __init__(self, forward, inverse):
        self.forward = forward
        self.inverse = inverse

    def __rshift__(self, other: "Bijection") -> "Bijection":
        """`self >> then` → apply `self` then `other` and inverses in the
        reverse order.

        """
        forward = lambda x: other.forward(self.forward(x))
        inverse = lambda y: self.inverse(other.inverse(y))
        return Bijection(forward, inverse)


def translation(t) -> Bijection:
    def forward(x):
        return x - t

    def inverse(y):
        return y + t

    return Bijection(forward, inverse)


def rotation(θ) -> Bijection:
    c = np.cos(θ)
    s = np.sin(θ)
    R = np.array([[c, -s], [s, c]])

    def forward(x):
        return x @ R

    def inverse(y):
        return y @ R.T

    return Bijection(forward, inverse)


def scale(α) -> Bijection:
    def forward(x):
        return α * x

    def inverse(y):
        return y / α

    return Bijection(forward, inverse)


def get_face_normalizer(landmarks) -> Bijection:
    eye_l = landmarks[slice(*LANDMARKS_INDICES["eye-l"])]
    eye_r = landmarks[slice(*LANDMARKS_INDICES["eye-r"])]

    eye_diff = np.mean(eye_r, axis=0) - np.mean(eye_l, axis=0)

    def get_center():
        return np.mean(landmarks[LIPS_SLICE], axis=0)

    def get_angle():
        dX, dY = eye_diff  # type: ignore
        return np.arctan2(dY, dX) - np.pi

    def get_scale():
        src_dist = np.linalg.norm(eye_diff)
        tgt_dist = 5
        return tgt_dist / src_dist

    t = get_center()
    θ = get_angle()
    α = get_scale()

    return translation(t) >> rotation(θ) >> scale(α)


# Utilities for getting normalized landmarks
def get_face_landmarks_npy_path(
    dataset, landmarks_type: Landmark, key, use_pca: bool = False
):
    suffix = "-pca" if use_pca else ""
    return os.path.join(
        "output",
        dataset.name,
        "face-landmarks-npy-" + landmarks_type + suffix,
        dataset.key_to_str(key) + ".npy",
    )


def get_first_face(landmarks: List[Any]) -> np.ndarray:
    try:
        return np.stack(landmarks[0])
    except IndexError:
        return np.zeros((NUM_LANDMARKS, 2)) + np.nan


def interpolate_missing_landmarks(xs):
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


def normalize_landmarks(landmarks):
    landmarks_norm = np.zeros(landmarks.shape)
    for i, landmarks1 in enumerate(landmarks):
        normalizer = get_face_normalizer(landmarks1)
        landmarks_norm[i] = normalizer.forward(landmarks1)
    return landmarks_norm


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
    landmarks1 = interpolate_missing_landmarks(landmarks1)
    landmarks1 = normalize_landmarks(landmarks1)
    landmarks1 = landmarks1[:, LIPS_SLICE]  # Extract lips
    landmarks1 = landmarks1.reshape(-1, LEN_LIPS * 2)

    if np.any(np.isnan(landmarks1)):
        print("WARN {} has NaNs".format(key_str))
        return

    if pca is not None:
        landmarks1 = pca.transform(landmarks1)

    make_folder(path_o)
    np.save(path_o, landmarks1)

    return key_str, path_o

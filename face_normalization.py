import pdb

import numpy as np

from constants import LANDMARKS_INDICES, LIPS_INDICES


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
        return np.mean(landmarks[slice(*LIPS_INDICES)], axis=0)

    def get_angle():
        dX, dY = eye_diff
        return np.arctan2(dY, dX) - np.pi

    def get_scale():
        src_dist = np.linalg.norm(eye_diff)
        tgt_dist = 5
        return tgt_dist / src_dist

    t = get_center()
    θ = get_angle()
    α = get_scale()

    return translation(t) >> rotation(θ) >> scale(α)

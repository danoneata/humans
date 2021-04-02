import json
import os
import pdb
import pickle
import time

from matplotlib import pyplot as plt

import numpy as np

import streamlit as st

from constants import LEN_LIPS
from scripts.obama.pca_landmarks import (
    load_pca,
    get_face_landmarks_npy_path,
    LANDMARKS_TYPE,
    Obama,
)


def draw_lips(ax, landmarks):
    kwargs = {"marker": "o"}
    ax.plot(landmarks[0:12, 0], landmarks[0:12, 1], **kwargs)
    ax.plot(landmarks[12:20, 0], landmarks[12:20, 1], **kwargs)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])


def main():
    pca = load_pca()

    st.markdown("## PCA mean")

    fig1, ax = plt.subplots()
    landmarks = pca.mean_.reshape(LEN_LIPS, 2)
    draw_lips(ax, landmarks)

    st.pyplot(fig1)

    st.markdown("## PCA components")

    fig2, axs = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
    landmarks = pca.components_.reshape(-1, LEN_LIPS, 2)

    for i, landmarks1 in enumerate(landmarks):
        row = i // 2
        col = i % 2
        draw_lips(axs[row, col], landmarks1)

    st.pyplot(fig2)

    st.markdown("## PCA reconstructions")

    dataset = Obama()

    keys = dataset.load_filelist("tiny")
    keys = keys[:5]

    fig3, axs = plt.subplots(nrows=5, ncols=3)

    for row, key in enumerate(keys):
        path = get_face_landmarks_npy_path(LANDMARKS_TYPE, key)

        landmarks_frames = np.load(path)
        num_landmarks = len(landmarks_frames)

        landmarks = landmarks_frames[num_landmarks // 2]
        coef = pca.transform(landmarks[np.newaxis])
        landmarks_reconstructed = pca.inverse_transform(coef)

        draw_lips(axs[row, 0], landmarks.reshape(LEN_LIPS, 2))
        draw_lips(axs[row, 1], landmarks_reconstructed.reshape(LEN_LIPS, 2))
        axs[row, 2].bar(np.arange(8), coef[0])

    st.pyplot(fig3)


if __name__ == "__main__":
    main()

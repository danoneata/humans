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


def draw_lips(ax, landmarks, plot_kwargs=None):
    if not plot_kwargs:
        plot_kwargs = {"marker": "o"}

    lips_inner = landmarks[:12]
    lips_outer = landmarks[12:]

    lips_inner = np.vstack((lips_inner, lips_inner[0]))
    lips_outer = np.vstack((lips_outer, lips_outer[0]))

    ax.plot(lips_inner[:, 0], lips_inner[:, 1], **plot_kwargs)
    ax.plot(lips_outer[:, 0], lips_outer[:, 1], **plot_kwargs)

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
    st.markdown("Showing the eight PCA components.")

    fig2, axs = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
    landmarks = pca.components_.reshape(-1, LEN_LIPS, 2)

    for i, landmarks1 in enumerate(landmarks):
        row = i // 2
        col = i % 2
        draw_lips(axs[row, col], landmarks1)

    st.pyplot(fig2)

    st.markdown("## PCA reconstructions")
    st.markdown("Inverting the PCA projection.")

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

        if row == 0:
            axs[row, 0].set_title("original")
            axs[row, 1].set_title("reconstructed")
            axs[row, 2].set_title("PCA coef")

    fig3.tight_layout()
    st.pyplot(fig3)

    st.markdown("## Predictions")

    keys = dataset.load_filelist("chunks-valid")
    keys = keys[:5]

    VIDEO_PRED_DIR = "output/obama/lips-predicted-baseline"

    for key in keys:
        st.video(os.path.join(VIDEO_PRED_DIR, key + ".mp4"))


    st.markdown("## Un-chunked video")
    st.video(os.path.join(VIDEO_PRED_DIR, "0tEHBVzZY4E-00.mp4"))

    # get_face_landmarks_npy_pred_path = lambda key: f"/home/doneata/src/espnet/egs2/obama/exp/baseline/output/lips/{key}.npy"
    # def get_face_landmarks_npy_true_path(key):
    #     *key, part = key.split("-")
    #     key = "-".join(key)
    #     return f"output/obama/face-landmarks-npy-dlib-pca-chunks/{key}/{part}.npy"

    # keys = dataset.load_filelist("chunks-test")
    # keys = keys[:5]

    # fig4, axs = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True)

    # for row, key in enumerate(keys):
    #     path_true = get_face_landmarks_npy_true_path(key)
    #     path_pred = get_face_landmarks_npy_pred_path(key)

    #     x_true = np.load(path_true)
    #     x_pred = np.load(path_pred).squeeze(0)

    #     i = len(x_true) // 2

    #     x_true = pca.inverse_transform(x_true[i: i + 1])
    #     x_pred = pca.inverse_transform(x_pred[i: i + 1])

    #     draw_lips(axs[row, 0], x_true.reshape(LEN_LIPS, 2))
    #     draw_lips(axs[row, 1], x_pred.reshape(LEN_LIPS, 2))

    # st.pyplot(fig4)


if __name__ == "__main__":
    main()

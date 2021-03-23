import json

from matplotlib import pyplot as plt

import cv2

import streamlit as st

from data import DATASETS
from scripts.extract_face_landmarks import iterate_frames
from constants import LANDMARKS_INDICES


LANDMARKS_COLORS = {
    "face": (0.682, 0.780, 0.909, 0.5),
    "eyebrow1": (1.0, 0.498, 0.055, 0.4),
    "eyebrow2": (1.0, 0.498, 0.055, 0.4),
    "nose": (0.345, 0.239, 0.443, 0.4),
    "nostril": (0.345, 0.239, 0.443, 0.4),
    "eye1": (0.596, 0.875, 0.541, 0.3),
    "eye2": (0.596, 0.875, 0.541, 0.3),
    "lips": (0.596, 0.875, 0.541, 0.3),
    "teeth": (0.596, 0.875, 0.541, 0.4),
}


def overlay_lips(image, landmarks):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for k, idxs in LANDMARKS_INDICES.items():
        xs, ys = zip(*landmarks[idxs])
        ax.plot(
            xs,
            ys,
            color=LANDMARKS_COLORS[k],
            marker="o",
            markersize=4,
            linestyle="-",
            linewidth=2,
        )
    ax.axis("off")
    ax.axis("scaled")
    return fig


def show1(dataset, key):
    video_path = dataset.get_video_path(key)
    landmarks_path = dataset.get_face_landmarks_path(key)

    with open(landmarks_path, "r") as f:
        landmarks = json.load(f)

    for frame, landmarks_frame in zip(iterate_frames(video_path), landmarks):
        # pick landmarks for the first (and only) face
        landmarks_face, *_ = landmarks_frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig = overlay_lips(frame_rgb, landmarks_face)
        # for testing purposes show results only for the first frame
        break

    # show
    st.markdown("Key: `{}`".format(key))
    st.video(video_path)
    st.pyplot(fig)
    st.markdown("---")


def main():
    dataset = DATASETS["obama-360p"]()
    keys = dataset.load_filelist("video-splits")
    for key in keys[:5]:
        show1(dataset, key)


if __name__ == "__main__":
    main()

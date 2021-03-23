import json
import os

from matplotlib import pyplot as plt

import cv2

import streamlit as st

from constants import LANDMARKS_INDICES
from data import Obama
from scripts.extract_face_landmarks import iterate_frames
from utils import normalize_face


LANDMARKS_TYPE = "dlib"
LANDMARKS_COLORS = {
    "face": (0.682, 0.780, 0.909, 0.5),
    "eyebrow-l": (1.0, 0.498, 0.055, 0.4),
    "eyebrow-r": (1.0, 0.498, 0.055, 0.4),
    "nose": (0.345, 0.239, 0.443, 0.4),
    "nostril": (0.345, 0.239, 0.443, 0.4),
    "eye-l": (0.596, 0.875, 0.541, 0.3),
    "eye-r": (0.596, 0.875, 0.541, 0.3),
    "lips": (0.596, 0.875, 0.541, 0.3),
    "teeth": (0.596, 0.875, 0.541, 0.4),
}


def overlay_landmarks(image, landmarks):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for k, (α, ω) in LANDMARKS_INDICES.items():
        xs, ys = zip(*landmarks[α: ω])
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
    dir_results = os.path.join("output", dataset.name, LANDMARKS_TYPE)
    video_norm_path = os.path.join(dir_results, key + "-norm.mp4")
    video_lips_path = os.path.join(dir_results, key + "-lips.mp4")
    landmarks_path = dataset.get_face_landmarks_path(key, landmark_type=LANDMARKS_TYPE)

    with open(landmarks_path, "r") as f:
        landmarks = json.load(f)

    for frame, landmarks_frame in zip(iterate_frames(video_path), landmarks):
        # pick landmarks for the first (and only) face
        landmarks_face, *_ = landmarks_frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig = overlay_landmarks(frame_rgb, landmarks_face)
        # for testing purposes show results only for the first frame
        break

    # show
    st.markdown("Key: `{}`".format(key))
    st.video(video_path)
    st.video(video_norm_path)
    st.video(video_lips_path)
    # st.pyplot(fig)
    st.markdown("---")


def main():
    dataset = Obama()
    keys = dataset.load_filelist("video-splits")
    for key in keys[:5]:
        show1(dataset, key)


if __name__ == "__main__":
    main()

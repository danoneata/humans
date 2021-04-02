import json
import os
import pdb
import subprocess

import cv2
import numpy as np
import torch

from torch.nn.functional import conv1d

from constants import LANDMARKS_INDICES
from data import Obama
from scripts.extract_face_landmarks import iterate_frames
from utils import normalize_face_mouth as normalize_face


def draw_lips(landmarks, size):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    poly = [
        landmarks[slice(*LANDMARKS_INDICES[t])].astype(int).reshape((-1, 1, 2))
        for t in ("lips", "teeth")
    ]
    cv2.polylines(img, poly, True, (255, 255, 255), 2)
    return img


def smooth_temp_landmarks(landmarks):
    landmarks = torch.tensor([landmarks_frame[0] for landmarks_frame in landmarks])
    # landmarks = torch.squeeze(landmarks, dim=1)  # first face in each frame

    landmarks = landmarks.float()
    N, D, _ = landmarks.shape
    w = torch.tensor([[[0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05]]]).float()
    # landmarks -- num_frames, num_landmarks, 2
    landmarks = torch.cat([landmarks[:3], landmarks, landmarks[-3:]])
    inp = landmarks.moveaxis(0, 2).reshape(D * 2, 1, N + 6)
    out = conv1d(inp, w)
    out = out.reshape(D, 2, N).moveaxis(2, 0)
    out = out.int()
    out = torch.unsqueeze(out, dim=1)
    return out.tolist()


def normalize_video(dataset, key):
    FPS = 29.97
    SIZE = 256, 128
    LANDMARKS_TYPE = "dlib"

    dir_out = os.path.join("output", dataset.name, LANDMARKS_TYPE)
    video_path_i = dataset.get_video_path(key)
    audio_path_i = os.path.join(dataset.base_path, "audio-split", key + ".wav")
    get_video_path = lambda suffix, ext: os.path.join(
        dir_out, key + "-" + suffix + "." + ext
    )
    landmarks_path = dataset.get_face_landmarks_path(key, landmark_type=LANDMARKS_TYPE)

    with open(landmarks_path, "r") as f:
        landmarks = json.load(f)

    os.makedirs(dir_out, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_norm = cv2.VideoWriter(get_video_path("norm", "avi"), fourcc, FPS, SIZE)
    video_lips = cv2.VideoWriter(get_video_path("lips", "avi"), fourcc, FPS, SIZE)

    frames = iterate_frames(video_path_i)
    for i, (frame, landmarks_frame) in enumerate(zip(frames, landmarks)):
        print("frame → {:5d}\r".format(i), end="")
        landmarks_face, *_ = landmarks_frame
        if i == 0:
            frame_norm, landmarks_face_norm, scale = normalize_face(
                frame, landmarks_face, SIZE
            )
        else:
            frame_norm, landmarks_face_norm, _ = normalize_face(
                frame, landmarks_face, SIZE, scale
            )
        frame_lips = draw_lips(landmarks_face_norm, SIZE)

        video_norm.write(frame_norm)
        video_lips.write(frame_lips)

    video_norm.release()
    video_lips.release()

    # reëncode video using x264 in order to be able to play in the browser
    # using streamlit
    for t in ("norm", "lips"):
        subprocess.run(
            # Q how to audio to the created video?!
            [
                "ffmpeg",
                "-y",
                "-i", get_video_path(t, "avi"),
                # "-i", audio_path_i,
                "-vcodec", "libx264",
                # "-acodec", "aac",
                # "-map", "0:v:0",
                # "-map", "1:a:0",
                get_video_path(t, "mp4"),
            ]
        )


def main():
    # TODO parameterize over dataset and filelist
    dataset = Obama()
    keys = dataset.load_filelist("video-splits")
    for key in keys[:5]:
        print("video →", key)
        normalize_video(dataset, key)


if __name__ == "__main__":
    main()

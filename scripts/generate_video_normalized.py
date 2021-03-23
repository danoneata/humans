import json
import os
import subprocess

import cv2

from data import Obama
from scripts.extract_face_landmarks import iterate_frames
from utils import normalize_face


def normalize_video(dataset, key):
    FPS = 29.97
    HEIGHT, WIDTH = 256, 256
    LANDMARKS_TYPE = "dlib"

    dir_out =os.path.join("output", dataset.name, LANDMARKS_TYPE) 
    video_path_i = dataset.get_video_path(key)
    video_path_o_avi = os.path.join(dir_out, key + ".avi")
    video_path_o_mp4 = os.path.join(dir_out, key + ".mp4")
    landmarks_path = dataset.get_face_landmarks_path(key, landmark_type=LANDMARKS_TYPE)

    with open(landmarks_path, "r") as f:
        landmarks = json.load(f)

    os.makedirs(dir_out, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path_o_avi, fourcc, FPS, (HEIGHT, WIDTH))

    frames = iterate_frames(video_path_i)
    for i, (frame, landmarks_frame) in enumerate(zip(frames, landmarks)):
        print("frame → {:5d}\r".format(i), end="")
        landmarks_face, *_ = landmarks_frame
        frame_norm = normalize_face(frame, landmarks_face)
        video.write(cv2.resize(frame_norm, (HEIGHT, WIDTH)))

    video.release()
    # reëncode video using x264 in order to be able to play in the browser
    # using streamlit
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path_o_avi, "-vcodec", "libx264", video_path_o_mp4]
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

import itertools
import json
import os
import pdb

import click
import cv2
import dlib


SHAPE_PREDICTOR_PATH = (
    "/home/doneata/src/dlib-models/shape_predictor_68_face_landmarks.dat"
)


def shape_to_list(shape):
    return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]


def detect_face_landmarks(detector, predictor, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    landmarks = [shape_to_list(predictor(gray, rect)) for rect in rects]
    return landmarks


def iterate_frames(path_video):
    video_capture = cv2.VideoCapture(path_video)
    while True:
        ret, image = video_capture.read()
        if not ret:
            break
        yield image


@click.command()
@click.option("--folder-video", "folder_video", type=click.Path(exists=True))
@click.option("--folder-landmarks", "folder_landmarks", type=click.Path(exists=True))
@click.option("--video-name", "video_name")
def main(folder_video, folder_landmarks, video_name):
    # load models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    name, ext = os.path.splitext(video_name)
    path_video = os.path.join(folder_video, video_name)
    path_out = os.path.join(folder_landmarks, name + ".json")

    # if os.path.exists(path_out):
    #     return

    landmarks = [
        detect_face_landmarks(detector, predictor, frame)
        for frame in iterate_frames(path_video)
    ]

    with open(path_out, "w") as f:
        json.dump(landmarks, f)


if __name__ == "__main__":
    main()

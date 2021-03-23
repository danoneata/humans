import itertools
import json
import os
import pdb
# import warnings

from functools import partial

from multiprocessing import Pool

import numpy as np

from typing import Any, List, Optional, Tuple

import click
import cv2
import dlib
import face_alignment

from data import DATASETS, Dataset


SHAPE_PREDICTOR_PATH = (
    "output/models/face-landmarks/dlib/shape_predictor_68_face_landmarks.dat"
)


def shape_to_list(shape):
    return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]


def detect_face_landmarks_lib(detector, predictor, image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(image_gray, 1)
    landmarks = [shape_to_list(predictor(image_gray, face)) for face in faces]
    return landmarks


def detect_face_alignment_fa(fa, image):
    res = fa.get_landmarks(image)
    if res:
        return [r.astype(int).tolist() for r in res]
    else:
        return []


def iterate_frames(path_video):
    video_capture = cv2.VideoCapture(path_video)
    while True:
        ret, image = video_capture.read()
        if not ret:
            break
        yield image


def make_folder(path):
    """Given a path to a file it creates the folder if it is missing."""
    folder, file_ = os.path.split(path)
    os.makedirs(folder, exist_ok=True)


def landmarks_to_numpy(landmarks: List[List[List[Tuple]]]) -> Optional[np.ndarray]:
    """The three nested `List`s iterate over (ⅰ) frames, (ⅱ) faces, and
    (ⅲ) landmarks. The `Tuple` corresponds to the xy coördinates of the 2D
    position of each landmark.

    Update: This function is currently not used because, surprisingly, storing
    the data as a NumPy array is more inefficient than storing it as JSON. The
    reason is that the coördinates are small numbers and they are better
    encoded by ASCII. See this StackOverflow answer for a more detailed
    discussion:

    https://stackoverflow.com/a/53408785/474311

    """
    # Exit early if faces are either missing or more than one is detected.
    have_one_face = all(len(landmarks_frame) == 1 for landmarks_frame in landmarks)
    if not have_one_face:
        return None
    # Select the landmarks for the first (and only) face in each frame.
    landmarks1: List[List[Tuple]] = [landmarks_frame[0] for landmarks_frame in landmarks]
    landmarks_np = np.stack(landmarks1)
    landmarks_np = landmarks_np.astype(int)
    return landmarks_np


def extract(dataset, detect_face_landmarks, landmark_type, key, verbose=0):
    # if os.path.exists(dataset.get_face_landmarks_path(key)):
    #     return
    if verbose > 0:
        print(dataset.key_to_str(key))
    landmarks = [
        detect_face_landmarks(frame)
        for frame in iterate_frames(dataset.get_video_path(key))
    ]
    out_path = dataset.get_face_landmarks_path(key, landmark_type)
    # landmarks_np = landmarks_to_numpy(landmarks)
    # if landmarks_np is not None:
    #     make_folder(out_path)
    #     np.save(out_path, landmarks_np)
    # else:
    #     warnings.warn("WARN Not all frames contain extactly one face " + key)
    make_folder(out_path)
    with open(out_path, "w") as f:
        json.dump(landmarks, f)


@click.command()
@click.option(
    "-d",
    "--dataset",
    "dataset_name",
    type=click.Choice(DATASETS),
    help="name of the dataset to work on",
)
@click.option("-f", "--filelist", help="specifies list of videos to process")
@click.option("--n-cpu", "n_cpu", default=1, help="number of cores to use")
@click.option("-v", "--verbose", default=0, count=True, help="how chatty to be")
@click.option("-lt", "--landmark_type", default="dlib",
              help="options: 1. dlib 2. face-alignment (see https://github.com/1adrianb/face-alignment)")
def main(dataset_name, filelist, landmark_type="dlib", n_cpu=1, verbose=0):
    dataset: Dataset = DATASETS[dataset_name]()
    keys = dataset.load_filelist(filelist)

    if landmark_type == "face-alignment":
        assert n_cpu == 1, "face-alignment extractor not implemented for parallel running"

    # Load face detection and alignment models
    if landmark_type == "dlib":
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        detect_face_landmarks = partial(detect_face_landmarks_lib, detector, predictor)
    elif landmark_type == "face-alignment":
        # TODO How to parallelize when running when CPU?
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cuda")
        detect_face_landmarks = partial(detect_face_alignment_fa, fa)
    else:
        assert False, "Unknown landmark type, please use 'face_landmarks' or 'face_alignment'"

    extract1 = partial(extract, dataset, detect_face_landmarks, landmark_type, verbose=verbose)

    if n_cpu == 1:
        for key in keys:
            extract1(key)
    elif n_cpu > 1:
        with Pool(n_cpu) as p:
            p.map(extract1, keys)
    else:
        assert False, "Invalid number of processes {}".format(n_cpu)


if __name__ == "__main__":
    main()


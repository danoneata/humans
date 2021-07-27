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

import torch

from data import DATASETS, Dataset
from utils import make_folder


SHAPE_PREDICTOR_PATH = (
    "output/models/face-landmarks/dlib/shape_predictor_68_face_landmarks.dat"
)

FACEMESH_MODEL_PATH = (
    "models/facemesh.pth"
)

# Indexes of FaceMeshpoints located around the lips
CONTOUR_POINTS_LIST = [0, 11, 12, 13, 14, 15, 16, 17, 18, 37, 38, 39, 40, 41, 42, 43, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 106, 146, 178, 179, 180, 181, 182, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 273, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 335, 375, 402, 403, 404, 405, 406, 407, 408, 409, 415]


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


def detect_face_landmarks_facemesh(detector, net, image):
    """ Returns the FaceMesh points located around the lip region
        of the image. It uses the DLIB face detector to crop the
        original image with a 25% margin around the face and
        feeds it into the FaceMesh code. The detection is a 
        set of 3D points. 
        Original code and model: https://github.com/thepowerfuldeez/facemesh.pytorch
    """
    # The increase in the area around the face detection
    # required by the FaceMesh net 
    percent = 0.25 
    # Image used by DLIB face detector
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = detector(image_gray)[0]
    l = face.left()
    t = face.top()
    w = face.right() - l
    h = face.bottom() - t
    # adjust top and left according to FaceMesh specs
    t1 = t - int(percent*h)
    l1 = l - int(percent*w)

    # Crop the image around the face with a `percent` margin
    img_o = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_img = img_o[t1:t1+int((1+2*percent)*h), l1:l1+int((1+2*percent)*w)]
    
    # Resize the image to 192x192
    img = cv2.resize(crop_img, (192, 192))
    detections = net.predict_on_image(img).cpu().numpy()
    # Return only the lip countour points normalised
    # to the original image size
    norm_factor = crop_img.shape[0]/192.0
    lip_detections = detections[CONTOUR_POINTS_LIST,:]*norm_factor
    return lip_detections.tolist()


def iterate_frames(path_video):
    video_capture = cv2.VideoCapture(path_video)
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path_video}")
    while True:
        ret, image = video_capture.read()
        if not ret:
            break
        yield image
    video_capture.release()


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
              help="options: 1. dlib 2. face-alignment (see https://github.com/1adrianb/face-alignment) 3. facemesh")
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
    elif landmark_type == "facemesh":
        from facemesh import FaceMesh
        gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net =  FaceMesh().to(gpu)
        net.load_weights(FACEMESH_MODEL_PATH)
        detector = dlib.get_frontal_face_detector()
        detect_face_landmarks = partial(detect_face_landmarks_facemesh, detector, net)
            
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


import itertools
import json
import os
import pdb

from abc import ABCMeta, abstractmethod

from functools import partial

from multiprocessing import Pool

from typing import Any, List

import click
import cv2
import dlib
import face_alignment
from skimage import io


SHAPE_PREDICTOR_PATH = (
    "output/models/face-landmarks/dlib/shape_predictor_68_face_landmarks.dat"
)


# Type aliases, mostly for readability purposes
Path = str
Key = Any


class Dataset(metaclass=ABCMeta):
    @abstractmethod
    def load_filelist(self, name: str) -> List[Key]:
        pass

    @abstractmethod
    def get_video_path(self, key: Key) -> Path:
        pass

    @abstractmethod
    def get_face_landmarks_path(self, key: Key, lt) -> Path:
        pass

    def key_to_str(self, key: Key) -> str:
        """Overload if you want to pretty print structured keys."""
        return str(Key)


class GRID(Dataset):
    """The GRID dataset, introduced in

    Cooke, Martin, et al. "An audio-visual corpus for speech perception and
    automatic speech recognition." The Journal of the Acoustical Society of
    America 120.5 (2006): 2421-2424.

    """

    base_path = "data/grid"
    video_ext = "mpg"

    folder_video = os.path.join(base_path, "video")
    folder_face_landmarks = os.path.join(base_path, "face-landmarks")

    def load_filelist(self, filelist):
        """The keys defined in the filelist are pairs of the type video name
        and speaker id; for example:

        bbaf4a s2
        pwwb3p s30
        bwwr1s s31

        """
        path = os.path.join(self.base_path, "filelists", filelist + ".txt")
        with open(path, "r") as f:
            return [line.strip().split() for line in f.readlines()]

    def get_video_path(self, key):
        """The videos are organized in subfolders based on the speaker id:
        
        data/grid/video/
        ├── s1
        │   ├── bbaf2n.mpg
        │   ├── bbaf3s.mpg
        │   └── ...
        ├── s2
        │   ├── bbaf1n.mpg
        │   ├── bbaf2s.mpg
        │   └── ...
        └── ...
        
        """
        video, speaker = key
        return os.path.join(self.folder_video, speaker, video + "." + self.video_ext)

    def get_face_landmarks_path(self, key, lt):
        """Use a folder structure similar to the one used for videos."""
        video, speaker = key
        return os.path.join(self.folder_face_landmarks, lt, speaker, video + ".json")

    def key_to_str(self, key):
        return " ".join(key)


DATASETS = {
    "grid": GRID,
}


def shape_to_list(shape):
    return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]


def detect_face_landmarks_lib(detector, predictor, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    landmarks = [shape_to_list(predictor(gray, rect)) for rect in rects]
    return landmarks


def detect_face_alignment_fa(fa, image):
    res = fa.get_landmarks(image)
    if res:
        return [r.tolist() for r in res]
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
              help="Options: 1. dlib 2. face-alignment (https://github.com/1adrianb/face-alignment)")
def main(dataset_name, filelist, landmark_type="face_landmarks", n_cpu=1, verbose=0):
    dataset: Dataset = DATASETS[dataset_name]()
    keys = dataset.load_filelist(filelist)

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
        assert False, "Unknown landmark type, please use \"face_landmarks\" or \"face_alignment\""

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


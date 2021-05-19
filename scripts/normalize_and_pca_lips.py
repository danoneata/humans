import click

from typing import List

import numpy as np

from tqdm import tqdm

from constants import Landmark
from data import DATASETS
from face_normalization import get_face_landmarks_npy_path, prepare_landmarks_npy
from scripts.obama.pca_landmarks import load_pca
from utils import make_folder

LANDMARKS_TYPE = "dlib"  # type: Landmark


@click.command()
@click.option(
    "-d",
    "--dataset",
    "dataset_name",
    required=True,
    type=click.Choice(list(DATASETS.keys())),
)
@click.option(
    "-f", "--filelist", required=True, help="specifies list of videos to process"
)
def main(dataset_name, filelist):
    dataset = DATASETS[dataset_name]()
    pca = load_pca(LANDMARKS_TYPE)

    keys = dataset.load_filelist(filelist)  # type: List[str]

    for key in tqdm(keys):
        path_i = get_face_landmarks_npy_path(dataset, LANDMARKS_TYPE, key)
        path_o = get_face_landmarks_npy_path(dataset, LANDMARKS_TYPE, key, use_pca=True)

        prepare_landmarks_npy(dataset, LANDMARKS_TYPE, key, overwrite=False)
        landmarks = np.load(path_i)
        landmarks = pca.transform(landmarks)

        make_folder(path_o)
        np.save(path_o, landmarks)


if __name__ == "__main__":
    main()

import os
import pdb

import click

from tqdm import tqdm

from data import Obama
from face_normalization import prepare_landmarks_npy
from utils import make_folder


ESPNET_EGS_PATH = os.path.expanduser("~/src/espnet/egs2/obama")
CURRENT_DIR = os.path.expanduser("~/work/humans")

DATASET = Obama()


def get_audio_path(key):
    return os.path.join(CURRENT_DIR, DATASET.base_path, "audio-split", key + ".wav")


def write_scp(path_scp, data_scp):
    make_folder(path_scp)
    with open(path_scp, "w") as f:
        for key, path in data_scp:
            f.write(key + " " + os.path.join(CURRENT_DIR, path) + "\n")


@click.command()
@click.option("-s", "--split")
@click.option("-l", "--landmarks-type", "landmarks_type")
@click.option("-w", "--overwrite", is_flag=True)
def main(split, landmarks_type, overwrite=False):
    keys = DATASET.load_filelist(split)

    data_lip_scp = [
        prepare_landmarks_npy(DATASET, landmarks_type, key, overwrite)
        for key in tqdm(keys)
    ]
    data_wav_scp = [(key, get_audio_path(key)) for key in keys]

    path_lip_scp = os.path.join(ESPNET_EGS_PATH, "data", split, "lip.scp")
    path_wav_scp = os.path.join(ESPNET_EGS_PATH, "data", split, "wav.scp")

    write_scp(path_wav_scp, data_wav_scp)
    write_scp(path_lip_scp, data_lip_scp)


if __name__ == "__main__":
    main()

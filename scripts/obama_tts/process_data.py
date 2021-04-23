import json
import os
import pdb
import subprocess

from collections import namedtuple

from typing import List

import click

from data import ObamaTTS
from face_normalization import prepare_landmarks_npy
from scripts.obama.pca_landmarks import load_pca
from scripts.obama.prepare_landmarks_espnet import write_scp


DATASET = ObamaTTS()
AUDIO_ORIG_DIR = os.path.join(DATASET.base_path, "audio-orig")


def resample_audio():
    for key in DATASET.load_filelist():
        src = os.path.join(AUDIO_ORIG_DIR, key + "." + DATASET.audio_ext)
        dst = DATASET.get_audio_path(key)
        # fmt: off
        subprocess.run(
            [
                "sox",
                src,
                "-b", "16",
                "-r", "16000",
                dst,
            ]
        )
        # fmt: on


def prepare_espnet_filelists():
    ESPNET_EGS_PATH = os.path.expanduser("~/src/espnet/egs2/obama")
    keys = DATASET.load_filelist()
    data_wav_scp = [(key, DATASET.get_audio_path(key)) for key in keys]
    path_wav_scp = os.path.join(ESPNET_EGS_PATH, "data", "obama-tts", "wav.scp")
    write_scp(path_wav_scp, data_wav_scp)


@click.command()
@click.option(
    "-t",
    "--todo",
    type=click.Choice(
        [
            "resample-audio",
            "prepare-espnet-filelists",
        ]
    ),
    required=True,
)
def main(todo, resolution=None):
    if todo == "resample-audio":
        resample_audio()
    elif todo == "prepare-espnet-filelists":
        prepare_espnet_filelists()
    else:
        assert False


if __name__ == "__main__":
    main()

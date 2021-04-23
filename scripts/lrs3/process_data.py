import json
import os
import pdb
import subprocess

from collections import namedtuple

from typing import List

import click

from data import LRS3
from face_normalization import prepare_landmarks_npy
from scripts.obama.pca_landmarks import load_pca
from scripts.obama.prepare_landmarks_espnet import write_scp


DATASET = LRS3()


def convert_video():
    for key in DATASET.load_filelist("test"):
        src = DATASET.get_video_path(key)
        dst = os.path.join("data/lrs3/video", DATASET.key_to_str(key) + "." + DATASET.video_ext)
        if os.path.exists(dst):
            continue
        # fmt: off
        subprocess.run(
            [
                "ffmpeg",
                "-i", src,
                "-c:v", "libx264",
                "-c:a", "aac",
                dst,
            ]
        )
        # fmt: on


def extract_audio():
    for key in DATASET.load_filelist("test"):
        src = DATASET.get_video_path(key)
        dst = DATASET.get_audio_path(key)
        # fmt: off
        subprocess.run(
            [
                "ffmpeg",
                "-i", src,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-acodec", "pcm_s16le",
                dst,
            ]
        )
        # fmt: on


def prepare_espnet_filelists():
    ESPNET_EGS_PATH = os.path.expanduser("~/src/espnet/egs2/obama")
    keys = DATASET.load_filelist("test")
    data_wav_scp = [
        (DATASET.key_to_str(key), dataset.get_audio_path(key)) for key in keys
    ]
    path_wav_scp = os.path.join(ESPNET_EGS_PATH, "data", "lrs3-test", "wav.scp")
    write_scp(path_wav_scp, data_wav_scp)


def normalize_landmarks(pca=None):
    for key in DATASET.load_filelist("test"):
        prepare_landmarks_npy(DATASET, "dlib", key, pca=pca, overwrite=True)


@click.command()
@click.option(
    "-t",
    "--todo",
    type=click.Choice(
        [
            "convert-video",
            "extract-audio",
            "prepare-espnet-filelists",
            "normalize-landmarks",
            "normalize-landmarks-pca",
        ]
    ),
)
def main(todo, resolution=None):
    if todo == "convert-video":
        convert_video()
    elif todo == "extract-audio":
        extract_audio()
    elif todo == "prepare-espnet-filelists":
        prepare_espnet_filelists()
    elif todo == "normalize-landmarks":
        normalize_landmarks()
    elif todo == "normalize-landmarks-pca":
        pca = load_pca()
        normalize_landmarks(pca=pca)
    else:
        assert False


if __name__ == "__main__":
    main()

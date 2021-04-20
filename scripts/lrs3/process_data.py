import json
import os
import pdb
import subprocess

from collections import namedtuple

from typing import List

import click

from data import LRS3 
from scripts.obama.prepare_landmarks_espnet import write_scp


def extract_audio():
    dataset = LRS3()
    for key in dataset.load_filelist("test"):
        src = dataset.get_video_path(key)
        dst = dataset.get_audio_path(key)
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
    dataset = LRS3()
    keys = dataset.load_filelist("test")
    data_wav_scp = [(dataset.key_to_str(key), dataset.get_audio_path(key)) for key in keys]
    path_wav_scp = os.path.join(ESPNET_EGS_PATH, "data", "lrs3-test", "wav.scp")
    write_scp(path_wav_scp, data_wav_scp)


@click.command()
@click.option(
    "-t",
    "--todo",
    type=click.Choice(["extract-audio", "prepare-espnet-filelists"]),
)
def main(todo, resolution=None):
    if todo == "extract-audio":
        extract_audio()
    elif todo == "prepare-espnet-filelists":
        prepare_espnet_filelists()
    else:
        assert False


if __name__ == "__main__":
    main()

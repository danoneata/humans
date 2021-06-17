"""Chunk audio and lips into smaller parts to fit in GPU memory when training
the audio-to-lip network with ESPnet.

"""
import os
import pdb

from itertools import count

from typing import List, Tuple

import click

import numpy as np

from pydub import AudioSegment

from toolz import concat, first

from data import DATASETS
from scripts.obama.prepare_landmarks_espnet import (
    CURRENT_DIR,
    ESPNET_EGS_PATH,
    write_scp,
)
from face_normalization import get_face_landmarks_npy_path
from utils import make_folder


# chunk characteristics
DURATION = 10_000  # ms
OVERLAP = 1_000  # ms
LANDMARKS_TYPE = "dlib"


def split_wav(dataset, key, verbose=0):
    path_i = dataset.get_audio_path(key)
    get_path_o = lambda key, i: os.path.join(
        CURRENT_DIR, "output", dataset.name, "audio-chunks", key, f"{i:03d}.wav"
    )

    audio = AudioSegment.from_wav(path_i)
    start = 0
    data_scp = []  # type: List[Tuple[str, str]]

    if verbose:
        print(key)
        print("total duration:", audio.duration_seconds)

    for i in count(0):
        if start >= len(audio) - OVERLAP:
            break

        path_o = get_path_o(key, i)
        make_folder(path_o)

        if not os.path.exists(path_o):
            chunk = audio[start : start + DURATION + OVERLAP]
            chunk.export(path_o, format="wav")

            if verbose:
                print(f"· {i:03d} {chunk.duration_seconds:6.2f}")

        start = start + DURATION

        key_chunk = key + "-" + f"{i:03d}"
        data_scp.append((key_chunk, path_o))

    if verbose:
        print()

    return data_scp


def split_lip(dataset, key, verbose, use_pca=True):
    path_i = get_face_landmarks_npy_path(dataset, LANDMARKS_TYPE, key, use_pca)
    suffix = "-pca" if use_pca else ""
    get_path_o = lambda key, i: os.path.join(
        CURRENT_DIR,
        "output",
        dataset.name,
        "face-landmarks-npy-" + LANDMARKS_TYPE + suffix + "-chunks",
        key,
        f"{i:03d}.npy",
    )

    lips = np.load(path_i)
    start = 0
    data_scp = []  # type: List[Tuple[str, str]]

    def time_to_frame(t):
        # time is in miliseconds
        return int(np.round(t / 1000 * dataset.fps))

    total_duration = len(lips) / dataset.fps
    total_duration_ms = 1000 * total_duration

    if verbose:
        print(key)
        print("total number of frames:", len(lips))
        print("total duration:        ", f"{total_duration:.2f}s")

    for i in count(0):
        if start >= total_duration_ms - OVERLAP:
            break

        path_o = get_path_o(key, i)
        make_folder(path_o)

        if not os.path.exists(path_o):
            α = time_to_frame(start)
            ω = time_to_frame(start + DURATION + OVERLAP)

            chunk = lips[α:ω]
            np.save(path_o, chunk)

            if verbose:
                print(f"· {i:03d} {len(chunk):3d} ← {α:4d} : {ω:4d}")

        start = start + DURATION

        key_chunk = key + "-" + f"{i:03d}"
        data_scp.append((key_chunk, path_o))

    if verbose:
        print()

    return data_scp


@click.command()
@click.option(
    "-d",
    "--dataset",
    "dataset_name",
    required=True,
    type=click.Choice(list(DATASETS.keys())),
)
@click.option("-s", "--split", required=True)
@click.option("-v", "--verbose", default=0, count=True, help="how chatty to be")
def main(dataset_name, split, verbose):
    dataset = DATASETS[dataset_name]()
    keys = dataset.load_filelist(split)

    _ = list(concat([split_lip(dataset, key, verbose, use_pca=False) for key in keys]))

    data_wav_scp = list(concat([split_wav(dataset, key, verbose) for key in keys]))
    data_lip_scp = list(concat([split_lip(dataset, key, verbose) for key in keys]))

    keys_wav = set(map(first, data_wav_scp))
    keys_lip = set(map(first, data_lip_scp))
    keys_common = keys_wav & keys_lip

    if len(keys_common) != len(keys_wav) or len(keys_common) != len(keys_lip):
        print("WARN Chunking the wavs and lips yielded different sized sets.")
        print("···· Keeping only the common keys")
        data_wav_scp = [datum for datum in data_wav_scp if first(datum) in keys_common]
        data_lip_scp = [datum for datum in data_lip_scp if first(datum) in keys_common]

    folder = dataset.name + "-chunks-" + split

    path_wav_scp = os.path.join(ESPNET_EGS_PATH, "data", folder, "wav.scp")
    path_lip_scp = os.path.join(ESPNET_EGS_PATH, "data", folder, "lip.scp")

    write_scp(path_wav_scp, data_wav_scp)
    write_scp(path_lip_scp, data_lip_scp)


if __name__ == "__main__":
    main()

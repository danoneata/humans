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

from scripts.obama.process_videos import FPS
from scripts.obama.prepare_landmarks_espnet import (
    CURRENT_DIR,
    DATASET,
    ESPNET_EGS_PATH,
    get_audio_path,
    get_face_landmarks_npy_path,
    write_scp,
)
from utils import make_folder


# chunk characteristics
DURATION = 10_000  # ms
OVERLAP = 1_000  # ms

LANDMARKS_TYPE = "dlib"
USE_PCA = True
SUFFIX = "-pca" if USE_PCA else ""

PATH_AUDIO = os.path.join(CURRENT_DIR, "output/obama/audio-chunks")
PATH_LIPS = os.path.join(
    CURRENT_DIR,
    "output/obama/face-landmarks-npy-" + LANDMARKS_TYPE + SUFFIX + "-chunks",
)


def split_wav(key, verbose=0):
    path_i = get_audio_path(key)
    get_path_o = lambda key, i: os.path.join(PATH_AUDIO, key, f"{i:03d}.wav")

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

        chunk = audio[start : start + DURATION + OVERLAP]
        chunk.export(path_o, format="wav")

        start = start + DURATION

        key_chunk = key + "-" + f"{i:03d}"
        data_scp.append((key_chunk, path_o))

        if verbose:
            print(f"· {i:03d} {chunk.duration_seconds:6.2f}")

    if verbose:
        print()

    return data_scp


def split_lip(key, verbose):
    path_i = get_face_landmarks_npy_path(LANDMARKS_TYPE, key, USE_PCA)
    get_path_o = lambda key, i: os.path.join(PATH_LIPS, key, f"{i:03d}.npy")

    lips = np.load(path_i)
    start = 0
    data_scp = []  # type: List[Tuple[str, str]]

    def time_to_frame(t):
        # time is in miliseconds
        return int(np.round(t / 1000 * FPS))

    total_duration = len(lips) / FPS
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

        α = time_to_frame(start)
        ω = time_to_frame(start + DURATION + OVERLAP)
        chunk = lips[α : ω]
        np.save(path_o, chunk)

        start = start + DURATION

        key_chunk = key + "-" + f"{i:03d}"
        data_scp.append((key_chunk, path_o))

        if verbose:
            print(f"· {i:03d} {len(chunk):3d} ← {α:4d} : {ω:4d}")

    if verbose:
        print()

    return data_scp


@click.command()
@click.option("-s", "--split", required=True)
@click.option("-v", "--verbose", default=0, count=True, help="how chatty to be")
def main(split, verbose):
    keys = DATASET.load_filelist(split)

    data_wav_scp = list(concat([split_wav(key, verbose) for key in keys]))
    data_lip_scp = list(concat([split_lip(key, verbose) for key in keys]))

    keys_wav = set(map(first, data_wav_scp))
    keys_lip = set(map(first, data_lip_scp))
    keys_common = keys_wav & keys_lip

    if len(keys_common) != len(keys_wav) or len(keys_common) != len(keys_lip):
        print("WARN Chunking the wavs and lips yielded different sized sets.")
        print("···· Keeping only the common keys")
        data_wav_scp = [datum for datum in data_wav_scp if first(datum) in keys_common]
        data_lip_scp = [datum for datum in data_lip_scp if first(datum) in keys_common]

    path_wav_scp = os.path.join(ESPNET_EGS_PATH, "data", "chunks-" + split, "wav.scp")
    path_lip_scp = os.path.join(ESPNET_EGS_PATH, "data", "chunks-" + split, "lip.scp")

    write_scp(path_wav_scp, data_wav_scp)
    write_scp(path_lip_scp, data_lip_scp)


if __name__ == "__main__":
    main()

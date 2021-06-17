import csv
import os
import pdb
import subprocess

from operator import itemgetter
from itertools import groupby

import click
import numpy as np
import streamlit as st

from matplotlib import pyplot as plt

from toolz import second

from scripts.extract_face_landmarks import iterate_frames


FPS = 29.97
τ = 2e-4


def cache(func, path, *args, **kwargs):
    if os.path.exists(path):
        return np.load(path)
    else:
        result = func(*args, **kwargs)
        np.save(path, result)
        return result


def get_color_histograms(path_video):
    return np.vstack(
        [
            np.hstack(
                [
                    np.histogram(frame[:, :, i].ravel(), 256, [0, 256], density=True)[0]
                    for i in [0, 1, 2]
                ]
            )
            for frame in iterate_frames(path_video)
        ]
    )

def get_frame_ranges(histograms, reference_time, τ, to_show=False):
    reference_frame = int(reference_time * FPS)  # reference frame
    dists = np.mean((histograms - histograms[reference_frame]) ** 2, axis=1)

    if to_show:
        fig, ax = plt.subplots()
        ax.hist(dists, bins=30)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(np.arange(len(is_trump)), is_trump)
        st.pyplot(fig)

    (is_trump_idxs,) = np.where(dists < τ)
    groups = groupby(enumerate(is_trump_idxs), lambda ix: ix[0] - ix[1])
    groups = (list(map(second, g)) for k, g in groups)

    ranges = ((group[0], group[-1]) for group in groups)
    ranges = (r for r in ranges if r[1] - r[0] > 3 * FPS)

    return list(ranges)


def split_videos(key, ranges, to_show=False):
    path_video = f"data/trump/video-360p/{key}.mp4"
    for i, (α, ω) in enumerate(ranges):
        start = α / FPS
        duration = (ω - α) / FPS

        dst = f"data/trump/video-split-360p/{key}-{i:03d}.mp4"
        if not os.path.exists(dst):
            # fmt: off
            subprocess.run(
                [
                    "ffmpeg",
                    "-ss", str(start),
                    "-i", path_video,
                    # Flag to overwrite destination file if it already exists.
                    # "-y",
                    "-t", str(duration),
                    # Reëncoding the video—it is slow, but also much more accurate
                    # than the `-c copy` option.
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    # "-filter:v", "fps=fps=" + str(FPS),
                    dst,
                ]
            )
            # fmt: on

        if to_show:
            st.video(dst)


def extract_audio(key, ranges, to_show=False):
    for i, _ in enumerate(ranges):
        src = f"data/trump/video-split-360p/{key}-{i:03d}.mp4"
        dst = f"data/trump/audio-split/{key}-{i:03d}.wav"

        if not os.path.exists(dst):
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

        if to_show:
            st.audio(dst)


def load_ranges(key):
    def to_frame_num(mm_ss):
        mm, ss = mm_ss.split(":")
        seconds = int(mm) * 60 + int(ss)
        return seconds * FPS
    VALID_POSES = ["Front", "Front-ish"]
    with open("data/trump/filelists/video-shots.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        ranges = [
            (to_frame_num(α), to_frame_num(ω))
            for k, α, ω, pose, _ in reader
            if k == key and pose in VALID_POSES
        ]
    return ranges


@click.command()
@click.option("-k", "--key", required=True)
@click.option("-t", "--reference-time", "reference_time", type=click.FLOAT, help="timestamp for the reference frame")
@click.option("-a", "--use-annotations", "to_use_annotations", help="use manual annotations of ranges", is_flag=True)
@click.option("--to-show", "to_show", is_flag=True)
def main(key, reference_time=None, to_use_annotations=False, to_show=False):
    assert reference_time is not None or to_use_annotations

    path_video = f"data/trump/video-360p/{key}.mp4"

    if to_use_annotations:
        ranges = load_ranges(key)
    else:
        path_hists = f"output/trump/video-360p/color-histograms-{key}.npy"
        path_ranges = f"output/trump/video-360p/frame-ranges-{key}-{τ}.npy"

        histograms = cache(get_color_histograms, path_hists, path_video)
        ranges = cache(get_frame_ranges, path_ranges, histograms, reference_time, τ, to_show)

    split_videos(key, ranges, to_show)
    extract_audio(key, ranges, to_show)


if __name__ == "__main__":
    main()

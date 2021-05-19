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


FPS = 30
τ = 2e-4


def cache(func, path, *args, **kwargs):
    if os.path.exists(path):
        return np.load(path)
    else:
        result = func(*args, **kwargs)
        np.save(path, result)
        return result


def get_color_histograms(path_video):
    return np.vstack([
        np.hstack([
            np.histogram(frame[:, :, i].ravel(), 256, [0, 256], density=True)[0]
            for i in [0, 1, 2]
        ])
        for frame in iterate_frames(path_video)
    ])


def get_frame_ranges(histograms, τ):
    R = 19 * 60 * FPS  # reference frame
    dists = np.mean((histograms - histograms[R]) ** 2, axis=1)

    # fig, ax = plt.subplots()
    # ax.hist(dists, bins=30)
    # st.pyplot(fig)

    # fig, ax = plt.subplots()
    # ax.plot(np.arange(len(is_trump)), is_trump)
    # st.pyplot(fig)

    is_trump_idxs, = np.where(dists < τ)
    groups = groupby(enumerate(is_trump_idxs), lambda ix: ix[0] - ix[1])
    groups = (list(map(second, g)) for k, g in groups)

    ranges = ((group[0], group[-1]) for group in groups)
    ranges = (r for r in ranges if r[1] - r[0] > 3 * FPS)

    return list(ranges)


@click.command()
@click.option("-k", "--key")
def main(key):
    path_video = os.path.join("data/trump/video-360p", key + ".mp4")
    path_hists = f"output/trump/video-360p/color-histograms-{key}.npy" 
    path_ranges = f"output/trump/video-360p/frame-ranges-{key}-{τ}.npy" 

    histograms = cache(get_color_histograms, path_hists, path_video)
    ranges = cache(get_frame_ranges, path_ranges, histograms, τ)
    
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
                    dst,
                ]
            )
            # fmt: on

        st.video(dst)


if __name__ == "__main__":
    main()

# Generates static web-page: http://www.zevo-tech.com/humans

import json
import os
import pdb
import random
import shutil
import subprocess
import sys

from copy import deepcopy

from itertools import groupby

import dominate

from dominate import tags
from dominate.util import raw

from matplotlib import pyplot as plt

import numpy as np

from tqdm import tqdm

from toolz import compose, first, partition_all, second

from data import DATASETS
from utils import make_folder
from scripts.generate_video_lips import (
    split_key,
    generate_pred_vs_true,
    load_pca,
    draw_lips,
    LEN_LIPS,
)
from scripts.iohannis.show_mean_lip import load_mean_lip


OUT_DIR = "output/www/diego"
TITLE = "Lip predictions on Diego"

dataset = DATASETS["diego-360p"]()
filelist_name = "full"

keys = dataset.load_filelist(filelist_name)


def load_segments(key):
    path = f"output/diego/speaking-segments/{key}.txt"
    with open(path, "r") as f:
        return [line.split() for line in f.readlines()]


def to_seconds(mm_ss):
    mm, ss = mm_ss.split(":")
    return int(mm) * 60 + int(ss)


def generate_true(dataset, split, key, to_overwrite=False, titles=None):
    video_path = os.path.join("output", "diego", "lips", "pred", key + ".mp4")

    if not to_overwrite and os.path.exists(video_path):
        return video_path

    pca = load_pca()

    DIR = "output/diego/face-landmarks-npy-predicted"
    landmarks = np.load(os.path.join(DIR, key + ".npy"))

    num_frames = len(landmarks)
    tmp_dir = os.path.join("/tmp", key)
    os.makedirs(tmp_dir, exist_ok=True)

    for i in tqdm(range(num_frames)):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2))

        draw_lips(ax, landmarks[i], dict(marker="o", color="blue"))
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.0, 2.0)
        ax.set_axis_off()

        path = os.path.join(tmp_dir, f"{i:05d}.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    make_folder(video_path)
    subprocess.run(
        [
            "ffmpeg",
            "-r",
            str(dataset.fps),
            "-i",
            os.path.join(tmp_dir, "%05d.png"),
            "-i",
            dataset.get_audio_path(key),
            "-y",
            "-c:v",
            "libx264",
            # "-vf", "fps=29.97",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            video_path,
        ]
    )

    shutil.rmtree(tmp_dir)
    return video_path


def copy(key, src_dir, dst_dir):
    EXT = ".mp4"
    src_path = os.path.join(src_dir, key + EXT)
    rel_path = os.path.join(dst_dir, key + EXT)
    dst_path = os.path.join(OUT_DIR, rel_path)
    make_folder(dst_path)
    # if not os.path.exists(dst_path):
    shutil.copy(src_path, dst_path)
    return rel_path


doc = dominate.document(title=TITLE)

with doc.head:
    tags.meta(**{"content": "text/html;charset=utf-8", "http-equiv": "Content-Type"})
    tags.meta(**{"content": "utf-8", "http-equiv": "encoding"})

    # jQuery
    tags.script(
        type="text/javascript",
        src="https://code.jquery.com/jquery-3.5.1.min.js",
    )
    # Bootstrap
    tags.link(
        rel="stylesheet",
        href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css",
    )
    tags.script(
        type="text/javascript",
        src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js",
        integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo",
        crossorigin="anonymous",
    )
    tags.script(
        type="text/javascript",
        src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js",
    )


with doc:
    with tags.body():
        with tags.div(cls="container"):
            tags.h1(TITLE, cls="mt-5")
            tags.p("""
            This web-page shows:
            ⅰ. videos of Diego;
            ⅱ. lip predictions based on the audio-to-lip network;
            ⅲ. manual annotations of the segments where only Diego is speaking.
            """)

            for key in keys:
                tags.h3(key)

                src_path = dataset.get_video_path(key)
                src_dir, _ = os.path.split(src_path)
                src = copy(key, src_dir=src_dir, dst_dir="data/video")

                with tags.div(cls="row"):
                    with tags.div(cls="col-sm-8"):
                        with tags.video(
                            controls=True, cls="embed-responsive", id=f"video-{key}"
                        ):
                            tags.source(src=src, type="video/mp4")

                    with tags.div(cls="col-sm-4"):

                        src_path = generate_true(
                            dataset,
                            filelist_name,
                            key,
                            to_overwrite=False,
                        )

                        src_dir, _ = os.path.split(src_path)
                        src = copy(key, src_dir=src_dir, dst_dir="data/lips")

                        tags.h4("lips")

                        with tags.video(
                            controls=True,
                            cls="embed-responsive",
                            id=f"lips-{key}",
                        ):
                            tags.source(src=src, type="video/mp4")

                        tags.h4("segments")

                        try:
                            segments = load_segments(key)
                        except FileNotFoundError:
                            segments = []

                        if segments:
                            with tags.ul(cls="overflow-auto", style="height: 200px"):
                                for segment in segments:
                                    with tags.li():
                                        tags.button(
                                            "play",
                                            type="button",
                                            cls="btn btn-light",
                                            data_start=to_seconds(segment[1]),
                                            data_end=to_seconds(segment[2]),
                                            data_video=key,
                                            onclick="playSegment(this)",
                                        )
                                        tags.code("{} – {}".format(segment[1], segment[2]))

                tags.hr()

    tags.script(type="text/javascript", src="main.js")

with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
    f.write(str(doc))

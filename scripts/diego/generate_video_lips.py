import os
import pdb
import shutil
import subprocess

import click

from matplotlib import pyplot as plt

import numpy as np

from tqdm import tqdm

from typing import Any

from constants import LEN_LIPS
from data import Diego
from scripts.show_lips_pca import draw_lips


DATASET = Diego("360p")
OUT_DIR = "output/diego-360p/video-lips-pca-reconstructed"

get_path_true = lambda key: "output/diego-360p/face-landmarks-npy-dlib-pca-reconstructed/{}.npy".format(key)


def generate_video_lips(key: Any):
    key_str = DATASET.key_to_str(key)
    video_path = os.path.join(OUT_DIR, key_str + ".mp4")

    # if os.path.exists(video_path):
    #     return

    landmarks_true_rec = np.load(get_path_true(key_str))

    tmp_dir = os.path.join("/tmp", key_str)
    os.makedirs(tmp_dir, exist_ok=True)

    for i, true_rec in enumerate(tqdm(landmarks_true_rec)):
        fig, ax = plt.subplots(figsize=(2, 2))

        draw_lips(ax, true_rec, dict(marker="o", color="blue"))

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.0, 2.0)
        ax.set_axis_off()

        path = os.path.join(tmp_dir, f"{i:05d}.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    subprocess.run(
        [
            "ffmpeg",
            "-r", str(DATASET.fps),
            "-i", os.path.join(tmp_dir, "%05d.png"),
            "-y",
            "-c:v", "libx264",
            # "-vf", "fps=29.97",
            "-pix_fmt", "yuv420p",
            video_path,
        ]
    )

    shutil.rmtree(tmp_dir)


TODO = {
    "video-lips": generate_video_lips,
}


@click.command()
@click.option("-t", "--todo", type=click.Choice(TODO.keys()), required=True)
def main(todo):
    keys = DATASET.load_filelist()
    os.makedirs(OUT_DIR, exist_ok=True)
    for key in keys:
        TODO[todo](key)


if __name__ == "__main__":
    main()

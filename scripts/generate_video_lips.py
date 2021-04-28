import os
import pickle
import pdb
import shutil
import subprocess

import click

from matplotlib import pyplot as plt

import numpy as np

from tqdm import tqdm


LEN_LIPS = 20


def draw_lips(ax, landmarks, plot_kwargs=None):
    if not plot_kwargs:
        plot_kwargs = {"marker": "o"}

    lips_inner = landmarks[:12]
    lips_outer = landmarks[12:]

    lips_inner = np.vstack((lips_inner, lips_inner[0]))
    lips_outer = np.vstack((lips_outer, lips_outer[0]))

    ax.plot(lips_inner[:, 0], lips_inner[:, 1], **plot_kwargs)
    ax.plot(lips_outer[:, 0], lips_outer[:, 1], **plot_kwargs)


@click.command()
@click.option(
    "-l",
    "--lips",
    "path_lips",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "-p",
    "--pca",
    "path_pca",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option("-o", "--output", "path_output", type=click.Path(), required=True)
@click.option(
    "-a",
    "--audio",
    "path_audio",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
)
@click.option("--fps", type=click.FLOAT, required=False, default=29.97)
def main(path_lips, path_pca, path_output, path_audio=None, fps=29.97):
    with open(path_pca, "rb") as f:
        pca = pickle.load(f)

    lips = np.load(path_lips)
    lips_rec = pca.inverse_transform(lips)
    lips_rec = lips_rec.reshape(-1, LEN_LIPS, 2)

    tmp_dir = ".tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    for i, lips in enumerate(tqdm(lips_rec)):
        fig, ax = plt.subplots(figsize=(2, 2))

        draw_lips(ax, lips, dict(marker="o", color="blue"))

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.0, 2.0)
        ax.set_axis_off()

        path = os.path.join(tmp_dir, f"{i:05d}.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    if path_audio:
        # fmt: off
        cmd = [
            "ffmpeg",
            "-r", str(fps),
            "-i", os.path.join(tmp_dir, "%05d.png"),
            "-i", path_audio,
            "-y",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            path_output,
        ]
        # fmt: on
    else:
        # fmt: off
        cmd = [
            "ffmpeg",
            "-r", str(fps),
            "-i", os.path.join(tmp_dir, "%05d.png"),
            "-y",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            path_output,
        ]
        # fmt: on

    subprocess.run(cmd)
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()

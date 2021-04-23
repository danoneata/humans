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
from data import LRS3
from evaluate import MODEL_DIR
from scripts.obama.pca_landmarks import load_pca
from scripts.show_lips_pca import draw_lips


DATASET = LRS3()
SPLIT = "test"
OUT_DIR = "output/lrs3/lips-predicted-baseline-obama"

get_path_true = lambda key: "output/lrs3/face-landmarks-npy-dlib-pca/{}.npy".format(key)


def get_path_pred(key: str, model: str):
    PATH = "/home/doneata/src/espnet/egs2/obama/exp"
    return os.path.join(PATH, MODEL_DIR[DATASET.name][model].format(SPLIT), "lips", key + ".npy")


def generate_pred_compare(key: Any):
    pca = load_pca()

    key_str = DATASET.key_to_str(key)
    video_path = os.path.join(OUT_DIR, "pred-compare", key_str + ".mp4")

    if os.path.exists(video_path):
        return

    def load_landmarks_pred(key_str, model):
        landmarks_pred_sm = np.load(get_path_pred(key_str, model))
        landmarks_pred_sm = landmarks_pred_sm.squeeze(0)
        landmarks_pred_lg = pca.inverse_transform(landmarks_pred_sm)
        landmarks_pred_lg = landmarks_pred_lg.reshape(-1, LEN_LIPS, 2)
        return landmarks_pred_lg

    landmarks_pred1_lg = load_landmarks_pred(key_str, "asr-ave")
    landmarks_pred2_lg = load_landmarks_pred(key_str, "asr-finetune-all-ave")

    try:
        landmarks_true_rec = pca.inverse_transform(np.load(get_path_true(key_str)))
        landmarks_true_rec = landmarks_true_rec.reshape(-1, LEN_LIPS, 2)
    except FileNotFoundError:
        landmarks_true_rec = [None] * len(landmarks_pred1_lg)

    tmp_dir = os.path.join("/tmp", key_str)
    os.makedirs(tmp_dir, exist_ok=True)

    for i, (true_rec, pred1_lg, pred2_lg) in enumerate(tqdm(zip(landmarks_true_rec, landmarks_pred1_lg, landmarks_pred2_lg))):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(2, 3.5))

        if true_rec is not None:
            draw_lips(axs[0], true_rec, dict(marker="o", color="blue"))
            draw_lips(axs[1], true_rec, dict(marker="o", color="blue"))

        draw_lips(axs[0], pred1_lg, dict(marker="o", color="orange"))
        draw_lips(axs[1], pred2_lg, dict(marker="o", color="orange"))

        for ax in axs:
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.0, 2.0)
            ax.set_axis_off()

        axs[0].set_title("method 1 (dec only)")
        axs[1].set_title("method 2 (enc + dec)")

        path = os.path.join(tmp_dir, f"{i:05d}.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    subprocess.run(
        [
            "ffmpeg",
            "-r", str(DATASET.fps),
            "-i", os.path.join(tmp_dir, "%05d.png"),
            "-i", DATASET.get_audio_path(key),
            "-y",
            "-c:v", "libx264",
            # "-vf", "fps=29.97",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            video_path,
        ]
    )

    shutil.rmtree(tmp_dir)


TODO = {
    "pred-compare": generate_pred_compare,
}


@click.command()
@click.option("-t", "--todo", type=click.Choice(TODO.keys()), required=True)
def main(todo):
    keys = DATASET.load_filelist(SPLIT)
    os.makedirs(os.path.join(OUT_DIR, todo), exist_ok=True)
    for key in keys[:64]:
        TODO[todo](key)


if __name__ == "__main__":
    main()

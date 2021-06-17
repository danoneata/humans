import os
import pickle
import pdb
import shutil
import subprocess

import click

from matplotlib import pyplot as plt

import numpy as np

from tqdm import tqdm

from evaluate import MODEL_DIR
from scripts.obama.pca_landmarks import load_pca
from utils import make_folder


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


def split_key(key):
    *keys1, part = key.split("-")
    key1 = "-".join(keys1)
    return key1, part


def get_path_true(dataset, key):
    key1, part = split_key(key)
    return os.path.join(
        "output", dataset.name, "face-landmarks-npy-dlib-chunks", key1, part + ".npy"
    )


def get_path_pred(dataset, key, model, split="test"):
    PATH = "/home/doneata/src/espnet/egs2/obama/exp"
    return os.path.join(PATH, MODEL_DIR[dataset.name_long][model].format(split), "lips", key + ".npy")


def get_path_audio(dataset, key):
    key1, part = split_key(key)
    return os.path.join("output", dataset.name, "audio-chunks", key1, part + ".wav")


def generate_true_vs_pred_multi(dataset, split, key, methods, to_overwrite=False, titles=None):
    video_path = os.path.join("output", dataset.name ,"lips-predicted-multiple-methods", "true", key + ".mp4")

    if not to_overwrite and os.path.exists(video_path):
        return video_path

    pca = load_pca()

    landmarks_true_org = np.load(get_path_true(dataset, key))
    landmarks_true_rec = pca.inverse_transform(pca.transform(landmarks_true_org))
    landmarks_true_org = landmarks_true_org.reshape(-1, LEN_LIPS, 2)
    landmarks_true_rec = landmarks_true_rec.reshape(-1, LEN_LIPS, 2)
    true = landmarks_true_rec

    def load_landmarks(path_pred):
        landmarks_pred_sm = np.load(path_pred)
        landmarks_pred_sm = landmarks_pred_sm.squeeze(0)
        landmarks_pred_lg = pca.inverse_transform(landmarks_pred_sm)
        landmarks_pred_lg = landmarks_pred_lg.reshape(-1, LEN_LIPS, 2)
        return landmarks_pred_lg

    tmp_dir = os.path.join("/tmp", key)
    os.makedirs(tmp_dir, exist_ok=True)

    num_frames = len(true)
    num_methods = len(methods)
    preds = {method: load_landmarks(get_path_pred(dataset, key, method, split)) for method in methods}

    for i in range(num_frames):
        fig, axs = plt.subplots(nrows=1, ncols=num_methods, figsize=(8.5, 2))

        for j, method in enumerate(methods):
            draw_lips(axs[j], true[i], dict(marker="o", color="blue"))
            draw_lips(axs[j], preds[method][i], dict(marker="o", color="orange"))
            if titles is None:
                axs[j].set_title(method)
            else:
                axs[j].set_title(titles[j])

        for ax in axs:
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
            "29.97",
            "-i",
            os.path.join(tmp_dir, "%05d.png"),
            "-i",
            get_path_audio(dataset, key),
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


def generate_true(dataset, split, key, to_overwrite=False):
    video_path = os.path.join("output", dataset.name ,"lips-predicted-baseline", "true", key + ".mp4")

    if not to_overwrite and os.path.exists(video_path):
        return video_path

    pca = load_pca()

    landmarks_true_org = np.load(get_path_true(dataset, key))
    landmarks_true_rec = pca.inverse_transform(pca.transform(landmarks_true_org))
    landmarks_true_org = landmarks_true_org.reshape(-1, LEN_LIPS, 2)
    landmarks_true_rec = landmarks_true_rec.reshape(-1, LEN_LIPS, 2)

    tmp_dir = os.path.join("/tmp", key)
    os.makedirs(tmp_dir, exist_ok=True)

    for i, (true_org, true_rec) in enumerate(
        tqdm(zip(landmarks_true_org, landmarks_true_rec))
    ):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4.50, 2.00))

        draw_lips(axs[0], true_org, dict(marker="o", color="blue"))
        draw_lips(axs[1], true_rec, dict(marker="o", color="blue"))

        for ax in axs:
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.0, 2.0)
            ax.set_axis_off()

        axs[0].set_title("true")
        axs[1].set_title("true (from pca)")

        path = os.path.join(tmp_dir, f"{i:05d}.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    make_folder(video_path)
    subprocess.run(
        [
            "ffmpeg",
            "-r",
            "29.97",
            "-i",
            os.path.join(tmp_dir, "%05d.png"),
            "-i",
            get_path_audio(dataset, key),
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


def generate_pred_vs_true(dataset, split, key, method, to_overwrite=False, pca=None):
    suffix = "" if not pca else ("pca-" + pca.name)
    video_path = os.path.join("output", dataset.name , "lips-predicted-baseline", "pred-vs-true" + suffix, key + ".mp4")

    if not to_overwrite and os.path.exists(video_path):
        return video_path

    if pca is None:
        pca = load_pca()

    landmarks_true_org = np.load(get_path_true(dataset, key))
    landmarks_true_rec = pca.inverse_transform(pca.transform(landmarks_true_org))
    landmarks_true_org = landmarks_true_org.reshape(-1, LEN_LIPS, 2)
    landmarks_true_rec = landmarks_true_rec.reshape(-1, LEN_LIPS, 2)

    landmarks_pred_sm = np.load(get_path_pred(dataset, key, method, split))
    landmarks_pred_sm = landmarks_pred_sm.squeeze(0)
    landmarks_pred_lg = pca.inverse_transform(landmarks_pred_sm)
    landmarks_pred_lg = landmarks_pred_lg.reshape(-1, LEN_LIPS, 2)

    tmp_dir = os.path.join("/tmp", key)
    os.makedirs(tmp_dir, exist_ok=True)

    for i, (true_org, true_rec, pred_lg) in enumerate(
        tqdm(zip(landmarks_true_org, landmarks_true_rec, landmarks_pred_lg))
    ):
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8.5, 2))

        draw_lips(axs[0], true_org, dict(marker="o", color="blue"))
        draw_lips(axs[1], true_rec, dict(marker="o", color="blue"))
        draw_lips(axs[2], pred_lg, dict(marker="o", color="orange"))
        draw_lips(axs[3], true_rec, dict(marker="o", color="blue"))
        draw_lips(axs[3], pred_lg, dict(marker="o", color="orange"))

        for ax in axs:
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.0, 2.0)
            ax.set_axis_off()

        axs[0].set_title("true")
        axs[1].set_title("true (from pca)")
        axs[2].set_title("pred")
        axs[3].set_title("true (from pca) vs. pred")

        path = os.path.join(tmp_dir, f"{i:05d}.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    make_folder(video_path)
    subprocess.run(
        [
            "ffmpeg",
            "-r",
            "29.97",
            "-i",
            os.path.join(tmp_dir, "%05d.png"),
            "-i",
            get_path_audio(dataset, key),
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

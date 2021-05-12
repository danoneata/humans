import os
import pdb
import shutil
import subprocess

import click

from matplotlib import pyplot as plt

import numpy as np

from tqdm import tqdm

from constants import LEN_LIPS
from data import Obama
from evaluate import MODEL_DIR
from scripts.obama.pca_landmarks import load_pca
from scripts.show_lips_pca import draw_lips



OUT_DIR = "output/obama/lips-predicted-baseline"
get_path_true = lambda key: "output/obama/face-landmarks-npy-dlib-chunks/{}/{}.npy".format(*split_key(key))
get_path_audio = lambda key: "output/obama/audio-chunks/{}/{}.wav".format(*split_key(key))


def get_path_pred(key, model):
    PATH = "/home/doneata/src/espnet/egs2/obama/exp"
    return os.path.join(PATH, MODEL_DIR[model].format(SPLIT), "lips", key + ".npy")


def split_key(key):
    *keys1, part = key.split("-")
    key1 = "-".join(keys1)
    return key1, part


def generate_true_long(key):
    """Generate using the un-chunked data."""
    path_l = "output/obama/face-landmarks-npy-dlib/{}.npy".format(key)
    path_a = "data/obama/audio-split/{}.wav".format(key)

    # landmarks_true = np.load(path_l)
    # landmarks_true = landmarks_true.reshape(-1, LEN_LIPS, 2)

    tmp_dir = os.path.join("/tmp", key)
    # os.makedirs(tmp_dir, exist_ok=True)

    # for i, true in enumerate(tqdm(landmarks_true)):
    #     fig, ax = plt.subplots() 

    #     draw_lips(ax, true, dict(marker="o", color="b"))

    #     ax.set_xlim(-2.5, 2.5)
    #     ax.set_ylim(-2.0, 2.0)

    #     path = os.path.join(tmp_dir, f"{i:05d}.png")
    #     plt.savefig(path)
    #     plt.close(fig)

    video_path = os.path.join(OUT_DIR, "true-long", key + ".mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-r", "29.97",
            "-i", os.path.join(tmp_dir, "%05d.png"),
            # "-i", path_a,
            "-y",
            "-c:v", "libx264",
            # "-vf", "fps=29.97",
            "-pix_fmt", "yuv420p",
            # "-c:a", "aac",
            # "-shortest",
            video_path,
        ]
    )


def generate_pred_vs_true(key):
    pca = load_pca()

    landmarks_true_org = np.load(get_path_true(key))
    landmarks_true_rec = pca.inverse_transform(pca.transform(landmarks_true_org))
    landmarks_true_org = landmarks_true_org.reshape(-1, LEN_LIPS, 2)
    landmarks_true_rec = landmarks_true_rec.reshape(-1, LEN_LIPS, 2)

    landmarks_pred_sm = np.load(get_path_pred(key, model="asr-ave"))
    landmarks_pred_sm = landmarks_pred_sm.squeeze(0)
    landmarks_pred_lg = pca.inverse_transform(landmarks_pred_sm)
    landmarks_pred_lg = landmarks_pred_lg.reshape(-1, LEN_LIPS, 2)

    tmp_dir = os.path.join("/tmp", key)
    os.makedirs(tmp_dir, exist_ok=True)

    for i, (true_org, true_rec, pred_lg) in enumerate(tqdm(zip(landmarks_true_org, landmarks_true_rec, landmarks_pred_lg))):
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

    video_path = os.path.join(OUT_DIR, "pred-vs-true", key + ".mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-r", "29.97",
            "-i", os.path.join(tmp_dir, "%05d.png"),
            "-i", get_path_audio(key),
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


def generate_pred_compare(key):
    pca = load_pca()

    landmarks_true_org = np.load(get_path_true(key))
    landmarks_true_rec = pca.inverse_transform(pca.transform(landmarks_true_org))
    landmarks_true_org = landmarks_true_org.reshape(-1, LEN_LIPS, 2)
    landmarks_true_rec = landmarks_true_rec.reshape(-1, LEN_LIPS, 2)

    def load_landmarks_pred(key, model):
        landmarks_pred_sm = np.load(get_path_pred(key, model))
        landmarks_pred_sm = landmarks_pred_sm.squeeze(0)
        landmarks_pred_lg = pca.inverse_transform(landmarks_pred_sm)
        landmarks_pred_lg = landmarks_pred_lg.reshape(-1, LEN_LIPS, 2)
        return landmarks_pred_lg

    landmarks_pred1_lg = load_landmarks_pred(key, "asr-ave")
    landmarks_pred2_lg = load_landmarks_pred(key, "asr-finetune-all-ave")

    tmp_dir = os.path.join("/tmp", key)
    os.makedirs(tmp_dir, exist_ok=True)

    for i, (true_rec, pred1_lg, pred2_lg) in enumerate(tqdm(zip(landmarks_true_org, landmarks_pred1_lg, landmarks_pred2_lg))):
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8.5, 2))

        draw_lips(axs[0], pred1_lg, dict(marker="o", color="orange"))
        draw_lips(axs[1], true_rec, dict(marker="o", color="blue"))
        draw_lips(axs[1], pred1_lg, dict(marker="o", color="orange"))
        draw_lips(axs[2], pred2_lg, dict(marker="o", color="orange"))
        draw_lips(axs[3], true_rec, dict(marker="o", color="blue"))
        draw_lips(axs[3], pred2_lg, dict(marker="o", color="orange"))

        for ax in axs:
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.0, 2.0)
            ax.set_axis_off()

        axs[0].set_title("pred 1")
        axs[1].set_title("true (from pca) vs. pred 1")
        axs[2].set_title("pred 2")
        axs[3].set_title("true (from pca) vs. pred 2")

        path = os.path.join(tmp_dir, f"{i:05d}.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    video_path = os.path.join(OUT_DIR, "pred-compare", key + ".mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-r", "29.97",
            "-i", os.path.join(tmp_dir, "%05d.png"),
            "-i", get_path_audio(key),
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


def generate_pred_compare_subsample(key):
    pca = load_pca()
    R = [1, 2, 4, 8, 16, 32]

    landmarks_true_org = np.load(get_path_true(key))
    landmarks_true_rec = pca.inverse_transform(pca.transform(landmarks_true_org))

    # reshape
    landmarks_true_org = landmarks_true_org.reshape(-1, LEN_LIPS, 2)
    landmarks_true_rec = landmarks_true_rec.reshape(-1, LEN_LIPS, 2)

    def get_path_pred(key, r, n):
        base_path = "/home/doneata/src/espnet/egs2/obama/exp"
        if r > 1:
            d = f"baseline-subsample/reciprocal-{r:02d}-num-{n:d}/asr"
        else:
            d = f"baseline/asr-finetune-all"
        return os.path.join(base_path, d, "output-test-ave", "lips", key + ".npy")

    def load_landmarks_pred(key, r, n):
        landmarks_pred_sm = np.load(get_path_pred(key, r, n))
        landmarks_pred_sm = landmarks_pred_sm.squeeze(0)
        landmarks_pred_lg = pca.inverse_transform(landmarks_pred_sm)
        landmarks_pred_lg = landmarks_pred_lg.reshape(-1, LEN_LIPS, 2)
        return landmarks_pred_lg

    landmarks_pred_all = {r: load_landmarks_pred(key, r, 0) for r in R}

    tmp_dir = os.path.join("/tmp", key)
    os.makedirs(tmp_dir, exist_ok=True)

    num_frames = len(landmarks_true_org)

    for i in enumerate(tqdm(range(num_frames))):
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8.5, 2))

        for r in R:
            row = r // 3
            col = r % 3

            draw_lips(axs[row, col], landmarks_pred_all[r][i], dict(marker="o", color="orange"))
            draw_lips(axs[row, col], landmarks_true_rec[i], dict(marker="o", color="blue"))
            axs[row, col].set_title(f"r = {r}")

            ax[row, col].set_xlim(-2.5, 2.5)
            ax[row, col].set_ylim(-2.0, 2.0)
            ax[row, col].set_axis_off()

        path = os.path.join(tmp_dir, f"{i:05d}.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    video_path = os.path.join(OUT_DIR, "pred-compare-subsample", key + ".mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-r", "29.97",
            "-i", os.path.join(tmp_dir, "%05d.png"),
            "-i", get_path_audio(key),
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
    "true-long": generate_true_long,
    "pred-vs-true": generate_pred_vs_true,
    "pred-compare": generate_pred_compare,
    "pred-compare-subsample": generate_pred_compare_subsample,
}


@click.command()
@click.option("-t", "--todo", type=click.Choice(TODO.keys()))
@click.option("-s", "--split", type=click.Choice(["valid", "test"]), required=True)
def main(todo, split):
    get_video_name = lambda key: key[:14]
    dataset = Obama()
    keys = dataset.load_filelist("chunks-" + split)
    video_names = sorted(list(set(map(get_video_name, keys))))
    selected_video_names = {
        "valid": {
            "41iHdxy7Kmg-00",
            "jrax-OJZrs0-00",
            "qnxYIhFfH-4-00",
            # "rnXk-uPmrz8-00",
            # "seIZB6qQEWY-00",
        },
        "test": {
            "GpdoHwhhCf0-00",
            "0bB0Cv4Oz0I-01",
        },
    }
    keys = [key for key in keys if get_video_name(key) in selected_video_names[split]]
    os.makedirs(os.path.join(OUT_DIR, todo), exist_ok=True)
    for key in keys:
        TODO[todo](key)


if __name__ == "__main__":
    main()

import pdb
import os

import click

from functools import partial

from typing import Any, Callable

import numpy as np

from sklearn.metrics import mean_squared_error  # type: ignore

from data import DATASETS, Dataset
from scripts.obama.pca_landmarks import load_pca


def split_key(key):
    *keys1, part = key.split("-")
    key1 = "-".join(keys1)
    return key1, part


MODEL_DIR = {
    "obama-360p": {
        "asr-ave": "baseline/asr/output-{}",
        "asr-finetune-all-best": "baseline/asr-finetune-all/output-{}-best",
        "asr-finetune-all-ave": "baseline/asr-finetune-all/output-{}-ave",
    },
    "lrs3": {
        "asr-ave": "baseline/asr/output-lrs3-{}-ave",
        "asr-finetune-all-ave": "baseline/asr-finetune-all/output-lrs3-{}-ave",
    },
}

for r in [2, 4, 8, 16, 32, 64]:
    for k in range(5):
        key = "subsample-reciprocal-{:02d}-num-{:d}".format(r, k)
        src = "baseline-subsample/reciprocal-{:02d}-num-{:d}/asr/output-test-ave".format(r, k)
        MODEL_DIR["obama-360p"][key] = src


FILELISTS = {
    "obama-360p": lambda split: "chunks-" + split,
    "lrs3": lambda split: split,
}


FACE_LANDMARKS_DIR = {
    "obama-360p": lambda key, use_pca: "output/obama/face-landmarks-npy-dlib{}-chunks/{}/{}.npy".format("-pca" if use_pca else "", *split_key(key)),
    "lrs3": lambda key, use_pca: "output/lrs3/face-landmarks-npy-dlib{}/{}.npy".format("-pca" if use_pca else "", key)
}


@click.command()
@click.option("-d", "--dataset", "dataset_name", type=click.Choice(DATASETS))
@click.option("-m", "--model", type=click.Choice(MODEL_DIR["obama-360p"]))
@click.option("-s", "--split", type=click.Choice(["valid", "test"]))
def main(dataset_name, model="asr-ave", split="test"):
    pca = load_pca()
    dataset = DATASETS[dataset_name]()  # type: Dataset

    get_path_true_lg = partial(FACE_LANDMARKS_DIR[dataset_name], use_pca=False)  # type: Callable[[Any], Any]
    get_path_true_sm = partial(FACE_LANDMARKS_DIR[dataset_name], use_pca=True)  # type: Callable[[Any], Any]
    get_path_pred_sm = lambda key: os.path.join(
        "/home/doneata/src/espnet/egs2/obama/exp",
        MODEL_DIR[dataset_name][model].format(split),
        "lips",
        key + ".npy",
    )

    filelist_name = FILELISTS[dataset_name](split)
    keys = [dataset.key_to_str(key) for key in dataset.load_filelist(filelist_name)]
    keys = [key for key in keys if os.path.exists(get_path_true_sm(key))]

    y_true_lg = [np.load(get_path_true_lg(key)) for key in keys]
    y_true_sm = [np.load(get_path_true_sm(key)) for key in keys]
    y_pred_sm = [np.load(get_path_pred_sm(key)).squeeze() for key in keys]

    # MSE in PCA space
    def mean_squared_error_zip(true, pred):
        n_true = len(true)
        n_pred = len(pred)
        i = min(n_true, n_pred)
        return mean_squared_error(true[:i], pred[:i])

    # err = [mean_squared_error(t, np.zeros(t.shape)) for t in y_true_sm]
    # print("MSE 8D:", np.mean(err))
    # pdb.set_trace()

    err = [mean_squared_error_zip(t, p) for t, p in zip(y_true_sm, y_pred_sm)]
    print("MSE  8D:", np.mean(err))

    # MSE in original space
    err = [mean_squared_error_zip(t, pca.inverse_transform(p)) for t, p in zip(y_true_lg, y_pred_sm)]
    print("MSE 20D:", np.mean(err))

    # MSE in original space
    err = [mean_squared_error_zip(t_lg, pca.inverse_transform(t_sm)) for t_lg, t_sm in zip(y_true_lg, y_true_sm)]
    print("MSE 20D:", np.mean(err))


if __name__ == "__main__":
    main()

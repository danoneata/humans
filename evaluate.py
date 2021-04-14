import pdb
import os

import click

import numpy as np

from sklearn.metrics import mean_squared_error  # type: ignore

from data import Obama
from scripts.obama.pca_landmarks import load_pca


def split_key(key):
    *keys1, part = key.split("-")
    key1 = "-".join(keys1)
    return key1, part


MODEL_DIR = {
    "asr-ave": "baseline/asr/output-{}",
    "asr-finetune-all-best": "baseline/asr-finetune-all/output-{}-best",
}


@click.command()
@click.option("-m", "--model", type=click.Choice(MODEL_DIR))
@click.option("-s", "--split", type=click.Choice(["valid", "test"]))
def main(model="asr-ave", split="test"):
    pca = load_pca()
    dataset = Obama()

    keys = dataset.load_filelist("chunks-" + split)

    get_path_true_lg = lambda key: "output/obama/face-landmarks-npy-dlib-chunks/{}/{}.npy".format(*split_key(key))
    get_path_true_sm = lambda key: "output/obama/face-landmarks-npy-dlib-pca-chunks/{}/{}.npy".format(*split_key(key))
    get_path_pred_sm = lambda key: os.path.join(
        "/home/doneata/src/espnet/egs2/obama/exp",
        MODEL_DIR[model].format(split),
        "lips",
        key + ".npy",
    )
 
    y_true_lg = [np.load(get_path_true_lg(key)) for key in keys]
    y_true_sm = [np.load(get_path_true_sm(key)) for key in keys]
    y_pred_sm = [np.load(get_path_pred_sm(key)).squeeze() for key in keys]

    # MSE in PCA space
    def mean_squared_error_zip(true, pred):
        n_true = len(true)
        n_pred = len(pred)
        i = min(n_true, n_pred)
        return mean_squared_error(true[:i], pred[:i])

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

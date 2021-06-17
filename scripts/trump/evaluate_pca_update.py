import pdb
import os

import numpy as np

from functools import partial

from evaluate import DATASETS, FILELISTS, FACE_LANDMARKS_DIR, MODEL_DIR, load_pca, mean_squared_error_zip
from scripts.iohannis.show_mean_lip import load_mean_lip


dataset_name = "trump-360p"
split = "chunks-corona-test"

dataset = DATASETS[dataset_name]()  # type: Dataset

filelist_name = FILELISTS[dataset_name](split)
keys = [dataset.key_to_str(key) for key in dataset.load_filelist(filelist_name)]
# keys = [key for key in keys if os.path.exists(get_path_true_sm(key))]

get_path_true_lg = partial(FACE_LANDMARKS_DIR[dataset_name], use_pca=False)  # type: Callable[[Any], Any]
get_path_true_sm = partial(FACE_LANDMARKS_DIR[dataset_name], use_pca=True)  # type: Callable[[Any], Any]

get_path_pred_sm = lambda model, key: os.path.join(
    "/home/doneata/src/espnet/egs2/obama/exp",
    MODEL_DIR[dataset_name][model].format(split),
    "lips",
    key + ".npy",
)

y_true_lg = [np.load(get_path_true_lg(key)) for key in keys]
y_true_sm = [np.load(get_path_true_sm(key)) for key in keys]

pca = load_pca()

p = pca.mean_[np.newaxis]
err = [mean_squared_error_zip(t, p) for t in y_true_lg]
print("mean obama:", np.mean(err))

p = load_mean_lip("trump-360p", "cpac").reshape(1, 40)
err = [mean_squared_error_zip(t, p) for t in y_true_lg]
print("mean trump (train):", np.mean(err))

p = load_mean_lip("trump-360p", "corona").reshape(1, 40)
err = [mean_squared_error_zip(t, p) for t in y_true_lg]
print("mean trump (test):", np.mean(err))

y_pred_sm = [np.load(get_path_pred_sm("asr-finetune-all-ave", key)).squeeze() for key in keys]
y_pred_lg = [pca.inverse_transform(y) for y in y_pred_sm]
err = [mean_squared_error_zip(t, p) for t, p in zip(y_true_lg, y_pred_lg)]
print("obama model:", np.mean(err))

pca.mean_ = load_mean_lip("trump-360p", "cpac").reshape(1, 40)
y_pred_sm = [np.load(get_path_pred_sm("asr-finetune-all-ave", key)).squeeze() for key in keys]
y_pred_lg = [pca.inverse_transform(y) for y in y_pred_sm]
err = [mean_squared_error_zip(t, p) for t, p in zip(y_true_lg, y_pred_lg)]
print("obama model + mean trump (train):", np.mean(err))

pca.mean_ = load_mean_lip("trump-360p", "corona").reshape(1, 40)
y_pred_sm = [np.load(get_path_pred_sm("asr-finetune-all-ave", key)).squeeze() for key in keys]
y_pred_lg = [pca.inverse_transform(y) for y in y_pred_sm]
err = [mean_squared_error_zip(t, p) for t, p in zip(y_true_lg, y_pred_lg)]
print("obama model + mean trump (test):", np.mean(err))

pca = load_pca()
for k in "030 060 120 240".split():
    y_pred_sm = [np.load(get_path_pred_sm(f"trump-chunks-cpac-ss-num-00{k}-seed-1337", key)).squeeze() for key in keys]
    y_pred_lg = [pca.inverse_transform(y) for y in y_pred_sm]
    err = [mean_squared_error_zip(t, p) for t, p in zip(y_true_lg, y_pred_lg)]
    print(f"trump model {k}:", np.mean(err))


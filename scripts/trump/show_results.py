import os

from itertools import groupby

import streamlit as st

from data import DATASETS
from scripts.generate_video_lips import generate_pred_vs_true, split_key

dataset = DATASETS["trump-360p"]()
split = "chunks-corona-test"

keys = dataset.load_filelist(split)
keys_grouped = groupby(keys, lambda key: split_key(key)[0])

i = 0
MAX_NUM_CHuNKS = 16

st.markdown(
    """This webpage presents the predicitons of the audio-to-lip network (pretrained on Obama) on a Trump video (his address regarding COVID19)."""
)


def get_pred_path(dataset, split, key):
    PATH = "/home/doneata/src/espnet/egs2/obama/exp"
    return os.path.join(
        PATH,
        "finetune",
        "trump-chunks-manual-shots-ss-num-00060-seed-0000",
        "asr",
        "predict-ave",
        f"trump-{split}",
        "lips",
        f"{key}.npy",
    )


def get_video_path(dataset, key):
    return os.path.join(
        "output",
        dataset.name,
        "lips-predicted-finetune-ss-num-00060-seed-0000",
        "pred-vs-true",
        key + ".mp4",
    )


for key1, keys_group in keys_grouped:
    st.markdown("### " + key1)
    st.video(dataset.get_video_path(key1))
    for key in keys_group:
        pred_path = get_pred_path(dataset, split, key)
        video_path = get_video_path(dataset, key)
        path_video_lips = generate_pred_vs_true(
            dataset, split, key, video_path, pred_path, to_overwrite=False
        )
        st.video(path_video_lips)
        if i > MAX_NUM_CHuNKS:
            break
        else:
            i += 1
    st.markdown("---")

from itertools import groupby

import streamlit as st

from data import DATASETS
from scripts.generate_video_lips import generate_pred_vs_true, split_key

dataset = DATASETS["trump-360p"]()
split = "chunks-corona"

keys = dataset.load_filelist(split)
keys_grouped = groupby(keys, lambda key: split_key(key)[0])

i = 0
MAX_NUM_CHuNKS = 16

for key1, keys_group in keys_grouped:
    st.markdown("### " + key1)
    st.video(dataset.get_video_path(key1))
    for key in keys_group:
        path_video_lips = generate_pred_vs_true(dataset, split, key)
        st.video(path_video_lips)
        if i > MAX_NUM_CHuNKS:
            break
        else:
            i += 1
    st.markdown("---")

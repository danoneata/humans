import os

import streamlit as st

from data import Diego


DATASET = Diego("360p")
DIR = "output/diego-360p/video-lips-pca-reconstructed"

for key in DATASET.load_filelist():
    video_path = os.path.join(DIR, key + ".mp4")
    st.video(DATASET.get_video_path(key))
    st.video(video_path)
    st.markdown("---")

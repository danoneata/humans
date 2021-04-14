# Generates static web-page: http://www.zevo-tech.com/humans

import json
import os
import pdb
import random
import shutil
import subprocess

from itertools import groupby

import dominate

from dominate import tags
from dominate.util import raw

from toolz import compose, first, partition_all, second

from data import Obama
from utils import make_folder


OUT_DIR = "output/www/baseline-obama"

dataset = Obama()
get_video_name = lambda key: key[:14]
keys = dataset.load_filelist("chunks-valid")
chunks = dict((key, list(group)) for key, group in groupby(keys, get_video_name))
video_splits = list(set(map(get_video_name, keys)))
selected_video_splits = [
    "41iHdxy7Kmg-00",
    "jrax-OJZrs0-00",
    "qnxYIhFfH-4-00",
    # "rnXk-uPmrz8-00",
    # "seIZB6qQEWY-00",
]

video_splits[:5]


def copy(key, src_dir, dst_dir):
    EXT = ".mp4"
    src_path = os.path.join(src_dir, key + EXT)
    rel_path = os.path.join(dst_dir, key + EXT)
    dst_path = os.path.join(OUT_DIR, rel_path)
    make_folder(dst_path)
    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)
    return rel_path


doc = dominate.document(title="Audio-to-lip conversion")

with doc.head:
    tags.meta(**{"content": "text/html;charset=utf-8", "http-equiv": "Content-Type"})
    tags.meta(**{"content": "utf-8", "http-equiv": "encoding"})

    # jQuery
    tags.script(
        type="text/javascript", src="https://code.jquery.com/jquery-3.5.1.min.js",
    )
    # Bootstrap
    tags.link(
        rel="stylesheet",
        href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css",
    )
    tags.script(
        type="text/javascript",
        src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js",
        integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo",
        crossorigin="anonymous",
    )
    tags.script(
        type="text/javascript",
        src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js",
    )


with doc:
    with tags.body():
        with tags.div(cls="container"):
            tags.h1("Audio-to-lip conversion", cls="mt-5")
            tags.p(
                """
                This webpages shows results of a Transformer audio-to-lip network on the Obama dataset.
                Each video shot was split into eleven-second chunks and with an overlap of one second;
                these short chunks are the input to the audio-to-lip network.
                We use two variants of the network:
                first, in which we fine-tune only the decoder part (the encoder is fixed based on a automatic speech recognition network),
                second, in which we fine-tune it entirely (both the encoder and the decoder);
                in terms of objective performance the first variant achieves 0.0275 MSE, while the second variant 0.0138 MSE (lower figures are better).
                Note that the groundtruth lips are reconstructred from the 8D PCA projection;
                similarly, the predictions are in the 8D PCA space and then inverted to the 20 × 2 coördinates using PCA.
                The lip videos show:
                (ⅰ) predictions based on the partially-tuned model;
                (ⅱ) comparison between the predictions from (ⅰ) and groundtruth;
                (ⅲ) predictions based on the fully-tuned model;
                (ⅳ) comparison between the predictions from (ⅱ) and groundtruth.
                In Firefox the lip videos can be played at 0.5× by right clicking on the video and selecting the "play speed" option.
                """
            )
            for video_split in selected_video_splits:
                tags.h3(video_split)
                src = copy(
                    video_split,
                    src_dir="data/obama/video-split",
                    dst_dir="data/video-split",
                )
                with tags.video(controls=True, cls="embed-responsive"):
                    tags.source(src=src, type="video/mp4")
                for chunk in chunks[video_split]:
                    src = copy(
                        chunk,
                        src_dir="output/obama/lips-predicted-baseline/pred-compare",
                        dst_dir="data/lips",
                    )
                    with tags.video(controls=True, cls="embed-responsive"):
                        tags.source(src=src, type="video/mp4")
                tags.hr()

    tags.script(type="text/javascript", src="main.js")

with open("output/www/baseline-obama/index.html", "w") as f:
    f.write(str(doc))

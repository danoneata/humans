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

from data import LRS3
from utils import make_folder


OUT_DIR = "output/www/baseline-obama-on-lrs3"

dataset = LRS3()
keys = dataset.load_filelist("test")
keys = keys[:64]


def copy(key, src_path, dst_dir):
    EXT = ".mp4"
    rel_path = os.path.join(dst_dir, key + EXT)
    dst_path = os.path.join(OUT_DIR, rel_path)
    make_folder(dst_path)
    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)
    return rel_path


doc = dominate.document(title="Evaluating the Obama model on LRS3")

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
            tags.h1("Evaluating the Obama model on LRS3", cls="mt-5")
            tags.p(
                """
                This webpage shows results of the Transformer audio-to-lip network trained on Obama dataset and evaluated on the LRS3 dataset.
                We show results for two variants:
                (ⅰ) train only the decoder part (and fix the encoder based on an ASR);
                (ⅱ) train both the encoder and decoder.
                The predictions are in orange and the groundtruth (the reconstructed lips from the 8D PCA representation) is in blue.
                """
            )
            for group in partition_all(4, keys):
                with tags.div(cls="row mb-2"):
                    for key in group:
                        key_str = dataset.key_to_str(key)
                        src_path = os.path.join("data/lrs3/video", key_str + ".mp4")
                        src_video = copy(
                            key_str,
                            # src_path=dataset.get_video_path(key),
                            src_path=src_path,
                            dst_dir="data/video",
                        )
                        src_path = os.path.join("output/lrs3/lips-predicted-baseline-obama/pred-compare", key_str + ".mp4")
                        src_lips = copy(
                            key_str,
                            src_path=src_path,
                            dst_dir="data/lips",
                        )
                        with tags.div(cls="col-md-3"):
                            tags.span(key_str, cls="text-monospace")
                            with tags.video(controls=True, cls="embed-responsive"):
                                tags.source(src=src_video)
                            with tags.video(controls=True, cls="embed-responsive"):
                                tags.source(src=src_lips)
                            # tags.hr()

    # tags.script(type="text/javascript", src="main.js")

with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
    f.write(str(doc))

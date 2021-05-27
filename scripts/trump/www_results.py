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

from data import DATASETS
from scripts.generate_video_lips import split_key
from utils import make_folder


OUT_DIR = "output/www/baseline-obama-on-trump"

dataset = DATASETS["trump-360p"]()
keys = dataset.load_filelist("chunks-test")


def copy(key, src_dir, dst_dir):
    EXT = ".mp4"
    src_path = os.path.join(src_dir, key + EXT)
    rel_path = os.path.join(dst_dir, key + EXT)
    dst_path = os.path.join(OUT_DIR, rel_path)
    make_folder(dst_path)
    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)
    return rel_path


doc = dominate.document(title="Evaluation Obama on Trump videos")

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
            tags.h1("Evaluating the Obama model on Trump videos", cls="mt-5")
            for key_shot, group in groupby(keys, lambda key: split_key(key)[0]):
                tags.h3(key_shot)
                src = copy(
                    key_shot,
                    src_dir=os.path.join(dataset.base_path, "video-split-" + dataset.video_res),
                    dst_dir="data/video-split",
                )
                with tags.video(controls=True, cls="embed-responsive"):
                    tags.source(src=src, type="video/mp4")
                for key_chunk in group:
                    src = copy(
                        key_chunk,
                        src_dir="output/trump/lips-predicted-baseline/pred-vs-true",
                        dst_dir="data/lips",
                    )
                    with tags.video(controls=True, cls="embed-responsive"):
                        tags.source(src=src, type="video/mp4")
                tags.hr()

    tags.script(type="text/javascript", src="main.js")

with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
    f.write(str(doc))

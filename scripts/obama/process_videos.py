import json
import os
import pdb
import subprocess

from collections import namedtuple

from typing import List

import click


# Folder from repo
# https://github.com/supasorn/synthesizing_obama_network_training/tree/master
BASE_PATH = "/home/doneata/src/synthesizing-obama-network-training/obama_data"
DATA_PATH = "data/obama"

FPS = 29.97
EXT = ".mp4"

VideoSplit = namedtuple("VideoSplit", "name part start_frame num_frames")


def get_video_split(folder) -> VideoSplit:
    name, part = folder.split("}}")
    with open(os.path.join(BASE_PATH, folder, "startframe.txt"), "r") as f:
        start_frame = int(f.read())
    with open(os.path.join(BASE_PATH, folder, "nframe.txt"), "r") as f:
        num_frames = int(f.read())
    return VideoSplit(name, part, start_frame, num_frames)


def get_video_splits() -> List[VideoSplit]:
    paths = os.listdir(BASE_PATH)
    folders = filter(lambda p: os.path.isdir(os.path.join(BASE_PATH, p)), paths)
    folders = filter(lambda f: f != "audio", folders)
    return list(map(get_video_split, sorted(folders)))


def save_video_splits():
    data = list(map(lambda v: v._asdict(), get_video_splits()))
    path = os.path.join(DATA_PATH, "filelists", "video-splits.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def split_videos():
    for video_split in get_video_splits():

        src = os.path.join(DATA_PATH, "videos", video_split.name + EXT)
        dst = os.path.join(
            DATA_PATH, "videos-split", video_split.name + "-" + video_split.part + EXT
        )

        start = str(video_split.start_frame / FPS)
        duration = str(video_split.num_frames / FPS)

        subprocess.run(
            [
                "ffmpeg",
                "-ss",
                start,
                "-i",
                src,
                "-t",
                duration,
                "-c",
                "copy",
                dst,
            ]
        )


def extract_audio():
    pass


@click.command()
@click.option(
    "-t",
    "--todo",
    type=click.Choice(["save-filelist", "split-videos", "extract-audio"]),
)
def main(todo):
    if todo == "save-filelist":
        save_video_splits()
    elif todo == "split-videos":
        split_videos()
    elif todo == "extract-audio":
        extract_audio()
    else:
        assert False


if __name__ == "__main__":
    main()

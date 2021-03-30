import json
import os
import pdb
import subprocess

from collections import namedtuple

from typing import List

import click


# Folder from repo
# https://github.com/supasorn/synthesizing_obama_network_training/tree/master
INFO_PATH = os.path.expanduser("~/src/synthesizing-obama-network-training")
INFO_PATH = os.environ.get("INFO_PATH", INFO_PATH)
DATA_PATH = "data/obama"

FPS = 29.97
EXT_VIDEO = ".mp4"
EXT_AUDIO = ".wav"

# Expected number of videos
NUM_VIDEOS = 301

# Video shots that do not contain Obama; see
# https://gitlab.com/zevo-tech/humans/-/issues/16
BAD_SPLITS = {"WjX0iJU3vtY-04", "WjX0iJU3vtY-06"}

VideoSplit = namedtuple("VideoSplit", "name part start_frame num_frames")


def get_video_split(folder) -> VideoSplit:
    name, part = folder.split("}}")
    path = os.path.join(INFO_PATH, "obama_data", folder)
    with open(os.path.join(path, "startframe.txt"), "r") as f:
        start_frame = int(f.read())
    with open(os.path.join(path, "nframe.txt"), "r") as f:
        num_frames = int(f.read())
    return VideoSplit(name, part, start_frame, num_frames)


def get_video_splits() -> List[VideoSplit]:
    folder = os.path.join(INFO_PATH, "obama_data")
    paths = os.listdir(folder)
    folders = filter(lambda p: os.path.isdir(os.path.join(folder, p)), paths)
    folders = filter(lambda f: f != "audio", folders)
    return list(map(get_video_split, sorted(folders)))


def save_video_splits():
    data = list(map(lambda v: v._asdict(), get_video_splits()))
    path = os.path.join(DATA_PATH, "filelists", "video-splits.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def split_videos():
    for video_split in get_video_splits():
        file_name_dst = video_split.name + "-" + video_split.part
        src = os.path.join(DATA_PATH, "video", video_split.name + EXT_VIDEO)
        dst = os.path.join(DATA_PATH, "video-split", file_name_dst + EXT_VIDEO)

        start = (video_split.start_frame - 1) / FPS
        duration = (video_split.num_frames - 1) / FPS

        # fmt: off
        subprocess.run(
            [
                "ffmpeg",
                "-ss", str(start),
                "-i", src,
                # Flag to overwrite destination file if it already exists.
                # "-y",
                "-t", str(duration),
                # Reëncoding the video—it is slow, but also much more accurate
                # than the `-c copy` option.
                "-c:v", "libx264",
                "-c:a", "aac",
                dst,
            ]
        )
        # fmt: on


def extract_audio():
    for video_split in get_video_splits():
        file_name = video_split.name + "-" + video_split.part
        src = os.path.join(DATA_PATH, "video-split", file_name + EXT_VIDEO)
        dst = os.path.join(DATA_PATH, "audio-split", file_name + EXT_AUDIO)
        # fmt: off
        subprocess.run(
            [
                "ffmpeg",
                "-i", src,
                "-vn",
                "-acodec", "copy",
                dst,
            ]
        )
        # fmt: on


def resize_videos(resolution):
    if resolution == "360p":
        size = "640x360"
    elif resolution == "480p":
        size = "854x480"
    else:
        assert False
    for video_split in get_video_splits():
        file_name = video_split.name + "-" + video_split.part
        src = os.path.join(DATA_PATH, "video-split", file_name + EXT_VIDEO)
        dst = os.path.join(DATA_PATH, "video-split-" + resolution, file_name + EXT_VIDEO)
        # fmt: off
        subprocess.run(
            [
                "ffmpeg",
                "-i", src,
                "-s", size,
                "-c:a", "copy",
                dst,
            ]
        )
        # fmt: on


@click.command()
@click.option(
    "-t",
    "--todo",
    type=click.Choice(["save-filelist", "split-videos", "extract-audio", "resize-videos"]),
)
@click.option(
    "-r",
    "--resolution",
    type=click.Choice(["360p", "480p"]),
)
def main(todo, resolution=None):
    if todo == "save-filelist":
        save_video_splits()
    elif todo == "split-videos":
        split_videos()
    elif todo == "extract-audio":
        extract_audio()
    elif todo == "resize-videos":
        resize_videos(resolution)
    else:
        assert False


if __name__ == "__main__":
    main()

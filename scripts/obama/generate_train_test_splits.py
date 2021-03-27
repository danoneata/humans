import os
import pdb
import random

from constants import SEED
from data import Obama
from scripts.obama.process_videos import BAD_SPLITS, NUM_VIDEOS


random.seed(SEED)


SPLIT_IDXS = {
    "train": (0, 250),
    "valid": (250, 275),
    "test": (275, 301),
}


def write_filelist(dataset, name, keys):
    path = os.path.join(dataset.base_path, "filelists", name + ".txt")
    with open(path, "w") as f:
        for key in keys:
            f.write(key)
            f.write("\n")


def main():
    dataset = Obama()

    keys = dataset.load_filelist("video-splits")
    keys = [key for key in keys if key not in BAD_SPLITS]

    get_video_name = lambda key: key[:11]
    video_names = list(set(map(get_video_name, keys)))

    # Check number of videos
    num_videos = len(video_names)
    err_message = "Expected {} videos, but got {} instead"
    assert num_videos == NUM_VIDEOS, err_message.format(NUM_VIDEOS, num_videos)

    random.shuffle(video_names)

    for s, idxs in SPLIT_IDXS.items():
        videos_selected = video_names[slice(*idxs)]
        keys_selected = [key for key in keys if get_video_name(key) in videos_selected]
        write_filelist(dataset, s, keys_selected)


if __name__ == "__main__":
    main()

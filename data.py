import os

from abc import ABCMeta, abstractmethod

from typing import Any, Dict, List, Optional, Tuple, Type

# Type aliases, mostly for readability purposes
Path = str
Key = Any


class Dataset(metaclass=ABCMeta):
    audio_ext = "wav"
    video_ext = "mp4"

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def base_path(self):
        return os.path.join("data", self.name)

    @abstractmethod
    def load_filelist(self, name: str) -> List[Key]:
        pass

    @abstractmethod
    def get_video_path(self, key: Key) -> Path:
        pass

    @abstractmethod
    def get_face_landmarks_path(self, key: Key, lt) -> Path:
        pass

    def key_to_str(self, key: Key) -> str:
        """Overload if you want to pretty print structured keys."""
        return str(key)


class GRID(Dataset):
    """The GRID dataset, introduced in

    Cooke, Martin, et al. "An audio-visual corpus for speech perception and
    automatic speech recognition." The Journal of the Acoustical Society of
    America 120.5 (2006): 2421-2424.

    """

    name = "grid"
    base_path = "data/grid"
    video_ext = "mpg"

    folder_video = os.path.join(base_path, "video")
    folder_face_landmarks = os.path.join(base_path, "face-landmarks")

    def load_filelist(self, filelist):
        """The keys defined in the filelist are pairs of the type video name
        and speaker id; for example:

        bbaf4a s2
        pwwb3p s30
        bwwr1s s31

        """
        path = os.path.join(self.base_path, "filelists", filelist + ".txt")
        with open(path, "r") as f:
            return [line.strip().split() for line in f.readlines()]

    def get_video_path(self, key):
        """The videos are organized in subfolders based on the speaker id:

        data/grid/video/
        ├── s1
        │   ├── bbaf2n.mpg
        │   ├── bbaf3s.mpg
        │   └── ...
        ├── s2
        │   ├── bbaf1n.mpg
        │   ├── bbaf2s.mpg
        │   └── ...
        └── ...

        """
        video, speaker = key
        return os.path.join(self.folder_video, speaker, video + "." + self.video_ext)

    def get_face_landmarks_path(self, key, lt):
        """Use a folder structure similar to the one used for videos."""
        video, speaker = key
        return os.path.join(self.folder_face_landmarks, lt, speaker, video + ".json")

    def key_to_str(self, key):
        return " ".join(key)


class Obama(Dataset):
    """Dataset containing Obama's weekly speeches. For more information see the
    README.md and this GitLab issue:

    https://gitlab.com/zevo-tech/humans/-/issues/11

    """

    name = "obama"
    base_path = "data/obama"
    video_ext = "mp4"
    audio_ext = "wav"
    fps = 29.97

    folder_video = os.path.join(base_path, "video-split-360p")
    # Path to face landmarks extracted by Cristi using `dlib`.
    folder_face_landmarks = os.path.join(base_path, "face-landmarks-360p")

    def load_filelist(self, filelist):
        path = os.path.join(self.base_path, "filelists", filelist + ".txt")
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_audio_path(self, key):
        return os.path.join(self.base_path, "audio-split", key + "." + self.audio_ext)

    # def load_filelist_json(self, filelist):
    #     path = os.path.join(self.base_path, "filelists", filelist + ".json")
    #     with open(path, "r") as f:
    #         data = json.load(f)
    #         return [datum["name"] + "-" + datum["part"] for datum in data]

    def get_video_path(self, key):
        return os.path.join(self.folder_video, key + "." + self.video_ext)

    def get_face_landmarks_path(self, key, landmark_type="dlib"):
        """Use a folder structure similar to the one used for videos."""
        return os.path.join(self.folder_face_landmarks, landmark_type, key + ".json")


class LRS3(Dataset):
    """Lip Reading Sentences 3: a dataset based on TED and TEDx videos.

    https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html

    """

    name = "lrs3"
    base_path = "data/lrs3"
    base_path_nas = "/mnt/private-share/speechDatabases/lrs3"
    video_ext = "mp4"
    fps = 25

    def load_filelist(self, filelist):
        """The keys defined in the filelist are tuples of the type
        video name, part number, split; for example:

        """
        path = os.path.join(self.base_path, "filelists", filelist + ".txt")
        with open(path, "r") as f:
            return [line.strip().split() for line in f.readlines()]

    def get_video_path(self, key):
        video, part, split = key
        return os.path.join(
            self.base_path_nas, split, video, part + "." + self.video_ext
        )

    def get_audio_path(self, key):
        audio_ext = "wav"
        video, part, _ = key
        return os.path.join(
            self.base_path, "audio", video + "-" + part + "." + audio_ext
        )

    def get_face_landmarks_path(self, key, landmark_type="dlib"):
        video, part, _ = key
        return os.path.join(
            self.base_path,
            "face-landmarks",
            landmark_type,
            video + "-" + part + ".json",
        )

    def key_to_str(self, key: Key) -> str:
        """Overload if you want to pretty print structured keys."""
        video, part, _ = key
        return video + "-" + part


class ObamaTTS(Dataset):
    name = "obama-tts"
    audio_ext = "wav"
    base_path = "data/obama-tts"
    fps = 29.97

    def load_filelist(self, name=""):
        return ["out_synth_{:03d}".format(i) for i in range(11)]

    def get_video_path(self, key):
        assert False

    def get_audio_path(self, key):
        return os.path.join(self.base_path, "audio", key + "." + self.audio_ext)

    def get_face_landmarks_path(self, key, landmark_type="dlib"):
        assert False


class Diego(Dataset):
    audio_ext = "wav"
    video_ext = "mp4"
    base_path = "data/diego"
    fps = 29.97
    name = "diego"

    def __init__(self, video_res):
        self.video_res = video_res
        self.name_long = "diego-" + video_res

    def load_filelist(self, name):
        path = os.path.join(self.base_path, "filelists", name + ".txt")
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_video_orig_path(self, key):
        return os.path.join(self.base_path, "video-orig", key + "." + self.video_ext)

    def get_video_path(self, key):
        return os.path.join(self.base_path, "video-" + self.video_res, key + "." + self.video_ext)

    def get_audio_path(self, key):
        return os.path.join(self.base_path, "audio", key + "." + self.audio_ext)

    def get_face_landmarks_path(self, key, landmark_type="dlib"):
        return os.path.join(self.base_path, "face-landmarks-" + self.video_res, key + ".json")


class Trump(Dataset):
    audio_ext = "wav"
    video_ext = "mp4"
    base_path = "data/trump"
    fps = 29.97
    name = "trump"

    def __init__(self, video_res):
        self.video_res = video_res
        self.name_long = "trump" + "-" + video_res

    def load_filelist(self, filelist):
        path = os.path.join(self.base_path, "filelists", filelist + ".txt")
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_video_orig_path(self, key):
        return os.path.join(self.base_path, "video-orig", key + "." + self.video_ext)

    def get_video_path(self, key):
        return os.path.join(self.base_path, "video-split-" + self.video_res, key + "." + self.video_ext)

    def get_audio_path(self, key):
        return os.path.join(self.base_path, "audio-split", key + "." + self.audio_ext)

    def get_face_landmarks_path(self, key, landmark_type="dlib"):
        return os.path.join(self.base_path, "face-landmarks-" + self.video_res, key + ".json")


class Iohannis(Dataset):
    name = "iohannis"

    def __init__(self, video_res):
        super().__init__()
        self.video_res = video_res
        self.fps = 25
        self.name_long = self.name + "-" + video_res

    def load_filelist(self, filelist):
        path = os.path.join(self.base_path, "filelists", filelist + ".txt")
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_video_orig_path(self, key):
        return os.path.join(self.base_path, "video-orig", key + "." + self.video_ext)

    def get_video_path(self, key):
        return os.path.join(self.base_path, "video-" + self.video_res, key + "." + self.video_ext)

    def get_audio_path(self, key):
        return os.path.join(self.base_path, "audio", key + "." + self.audio_ext)

    def get_face_landmarks_path(self, key, landmark_type="dlib"):
        return os.path.join(self.base_path, "face-landmarks-" + self.video_res, key + ".json")


DATASETS = {
    "grid": GRID,
    # TODO Parameterize dataset by video size.
    "obama-360p": Obama,
    "lrs3": LRS3,
    "obama-tts": ObamaTTS,
    "diego-360p": lambda: Diego(video_res="360p"),
    "diego-1080p": lambda: Diego(video_res="1080p"),
    "trump-360p": lambda: Trump(video_res="360p"),
    "iohannis-360p": lambda: Iohannis(video_res="360p"),
    "iohannis-720p": lambda: Iohannis(video_res="720p"),
}

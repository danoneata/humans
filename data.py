import os

from abc import ABCMeta, abstractmethod

from typing import Any, List, Optional, Tuple

# Type aliases, mostly for readability purposes
Path = str
Key = Any


class Dataset(metaclass=ABCMeta):
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
        return str(Key)


class GRID(Dataset):
    """The GRID dataset, introduced in

    Cooke, Martin, et al. "An audio-visual corpus for speech perception and
    automatic speech recognition." The Journal of the Acoustical Society of
    America 120.5 (2006): 2421-2424.

    """

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

    base_path = "data/obama"
    video_ext = "mp4"

    folder_video = os.path.join(base_path, "video-split-360p")
    # Path to face landmarks extracted by Cristi using `dlib`.
    folder_face_landmarks = os.path.join(base_path, "face-landmarks-360p")

    def load_filelist(self, filelist):
        path = os.path.join(self.base_path, "filelists", filelist + ".txt")
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_video_path(self, key):
        return os.path.join(self.folder_video, key + "." + self.video_ext)

    def get_face_landmarks_path(self, key, landmark_type="dlib"):
        """Use a folder structure similar to the one used for videos."""
        return os.path.join(self.folder_face_landmarks, landmark_type, key + ".json")


DATASETS = {
    "grid": GRID,
    # TODO Parameterize dataset by video size.
    "obama-360p": Obama,
}

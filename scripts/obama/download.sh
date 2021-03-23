n_jobs=12

base_url="https://www.youtube.com/watch?v="
base_path="data/obama"

# The filelist video-names.txt is obtained from the following repository:
# https://github.com/supasorn/synthesizing_obama_network_training/tree/master/obama_data
filelist="${base_path}/filelists/video-names.txt"
video_dir="${base_path}/videos"
subtitles_dir="${base_path}/subtitles"

mkdir -p ${video_dir}

parallel --bar -a $filelist -j $n_jobs youtube-dl ${base_url}{} -f best -o "${video_dir}/%\(id\)s.%\(ext\)s"
# Separate command for historical reasons—I've ran the download in two separate
# steps—but the two can certainly be combined.
parallel --bar -a $filelist -j $n_jobs youtube-dl ${base_url}{} --write-sub --skip-download -o "${subtitles_dir}/%\(id\)s.%\(ext\)s"

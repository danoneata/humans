n_jobs=12

base_url="https://www.youtube.com/watch?v="
base_path="data/obama"

# The filelist video-names.txt is obtained from the following repository:
# https://github.com/supasorn/synthesizing_obama_network_training/tree/master/obama_data
filelist="${base_path}/filelists/video-names.txt"
video_dir="${base_path}/videos"

mkdir -p ${video_dir}

# If there are problems with some videos use `-f best` option for youtube-dl
parallel --bar -a $filelist -j $n_jobs youtube-dl ${base_url}{} -o "${video_dir}/%\(id\)s.%\(ext\)s"

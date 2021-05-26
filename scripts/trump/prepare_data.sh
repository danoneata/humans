# keys:
# - KJTlo4bQL5c → 19 * 60
# - xrPZBTNjX_o → 20
# - KJTlo4bQL5c
# - QVXtNkzeKUU
# - H-DSfvYCKwY
# - FdAh2HJ98WE
# see: https://gitlab.com/zevo-tech/humans/-/issues/29
key=$1
ref=$2
if [ -z $key ]; then
    echo "ERROR: Missing key"
    exit 1;
fi
if [ -z $ref ]; then
    echo "ERROR: Missing reference time (in seconds)"
    exit 1;
fi
# youtube-dl https://www.youtube.com/watch?v=${key} -f best -o data/trump/video-orig/${key}.mp4
# youtube-dl https://www.youtube.com/watch?v=${key} --write-auto-sub --skip-download -o data/trump/subtitles/${key}
# ffmpeg -i data/trump/video-orig/${key}.mp4 -s 640x360 -c:a copy data/trump/video-360p/${key}.mp4
python scripts/trump/split_video.py -k $key -r $ref

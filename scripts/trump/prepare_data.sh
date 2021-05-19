key=KJTlo4bQL5c 
# youtube-dl https://www.youtube.com/watch?v=${key} -f best -o data/trump/video-orig/${key}.mp4
# youtube-dl https://www.youtube.com/watch?v=${key} --write-auto-sub --skip-download -o data/trump/subtitles/${key}
# ffmpeg -i data/trump/video-orig/${key}.mp4 -s 640x360 -c:a copy data/trump/video-360p/${key}.mp4
python data/trump/split_video.py $key

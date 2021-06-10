key=WBmIwnH7HRk
mkdir -p data/iohannis/{audio,video-orig,video-360p}
youtube-dl https://www.youtube.com/watch?v=${key} -f best -o data/iohannis/video-orig/${key}.mp4
ffmpeg -i data/iohannis/video-orig/${key}.mp4 -s 640x360 -c:a copy data/iohannis/video-360p/${key}.mp4
ffmpeg -i data/iohannis/video-orig/${key}.mp4 -vn -ac 1 -ar 16000 -acodec pcm_s16le data/iohannis/audio/${key}.wav

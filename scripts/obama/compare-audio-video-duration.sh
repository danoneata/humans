get_duration () {
    path=$1
    ffprobe \
        -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        ${path}
}

cat data/obama/filelists/video-splits.txt | \
while read key; do
    audio_dur=$(get_duration data/obama/audio-split/${key}.wav)
    video_dur=$(get_duration data/obama/video-split/${key}.mp4)
    diff=$(echo "scale=3; $video_dur - $audio_dur" | bc)
    printf "%s %6.2f %6.2f %.3f\n" $key $audio_dur $video_dur $diff
done | tee /tmp/audio-video.txt

get_duration () {
    path=$1
    ffprobe \
        -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        ${path}
}

FPS=29.97

cat data/obama/filelists/video-splits.txt | \
while read key; do
    # durationa—audio and video
    audio_dur=$(get_duration data/obama/audio-split/${key}.wav)
    video_dur=$(get_duration data/obama/video-split/${key}.mp4)
    diff_dur=$(echo "scale=3; $video_dur - $audio_dur" | bc)
    # number of frames—lip landmarks and calculated based on the duration
    num_frames_lips=$(cat data/obama/face-landmarks-360p/dlib/${key}.json | jq 'length')
    num_frames_calc=$(echo "scale=3; $video_dur * $FPS" | bc)
    diff_frames=$(echo "scale=3; $num_frames_calc - $num_frames_lips" | bc)
    printf "%s %6.2f %6.2f %.3f %5d %7.1f %+3.1f\n" \
        $key \
        $audio_dur $video_dur $diff_dur \
        $num_frames_lips $num_frames_calc $diff_frames
done | tee /tmp/audio-video.txt

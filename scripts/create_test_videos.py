import sys, os, json
import cv2
import numpy as np
from scripts.extract_face_landmarks import iterate_frames
import matplotlib.pylab as plt


method_folders = {"1":"dlib", "2":"face-alignment", "3":"facemesh"}

def overlay_landmarks(image, landmarks, folder, vid, i):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], zorder=2, s=0.5, color="lime")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("video_output/" + folder + "/" + vid + "_" + str(i).zfill(4) + ".jpg")
    plt.close()


def main(video_path, landmarks_path, method):

    with open(landmarks_path, "r") as f:
        landmarks = json.load(f)
        landmarks = np.array(np.squeeze(landmarks))
        print(landmarks.shape)

    vid = os.path.splitext(os.path.basename(video_path))[0]
    folder = method_folders[method]

    if not os.path.exists(os.path.join("video_output", folder)):
        os.makedirs(os.path.join("video_output", folder))

    i = 0
    for frame, landmarks_frame in zip(iterate_frames(video_path), landmarks):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay_landmarks(frame_rgb, landmarks_frame, folder, vid, i)
        if i == 10:
            break
        i += 1

    cmd = (
        "cat video_output/"
        + folder
        + "/"
        + vid
        + "*.jpg | ffmpeg -f image2pipe -r 10 -vcodec mjpeg -i - -vcodec libx264 video_output/out_"
        + folder
        + "_"
        + vid
        + ".mp4"
    )
    os.system(cmd)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(
            "Usage python create_test_videos <video_path> <landmarks_path> <method {1:dlib,2:face-aligment,3:facemesh}>"
        )

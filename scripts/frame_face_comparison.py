import os
import json
import sys


def load_json_file(file_path):
    with open(file_path) as json_file:
        json_data = json.load(json_file)
        return json_data


def load_filelist(filelist):
    with open(filelist, "r") as f:
        return [line.strip().split() for line in f.readlines()]


def main():
    # input_filelist = "/home/speed-marius/Speed/Humans/face-landmarks/listOfFa.list"
    input_filelist = "/home/speed-marius/Speed/Humans/face-landmarks/listOfDlib.list"
    # input_filelist = "/home/speed-marius/Speed/Humans/face-landmarks/test_list.txt"
    output_file = '/home/speed-marius/Speed/Humans/face_frame_output.txt'
    # Erase existing output file
    open(output_file, 'w').close()
    list_of_landmarks_json = load_filelist(input_filelist)
    # print(list_of_landmarks_json)
    for landmarks_file in list_of_landmarks_json:
        landmarks_file = landmarks_file[0]
        # print(landmarks_file)
        filename = landmarks_file[str(landmarks_file).rfind('/') + 1:]
        print(filename)
        json_file_data = load_json_file(landmarks_file)

        # Test printing
        # print('No. of JSONS/Frames:')
        # print(len(json_file_data))
        # print('1st array/ no. of faces:')
        # print(json_file_data[1])
        # print(len(json_file_data[1]))
        # print('2nd array/ no. of keypoints:')
        # print(json_file_data[0][0])
        # print(len(json_file_data[0][0]))
        # print('3rd array/ first tuple (keypoint) of landmarks:')
        # print(json_file_data[0][0][0])
        # print(len(json_file_data[0][0][0]))

        landmarks_all = json_file_data
        # problem_landmarks = []
        with open(output_file, 'a+') as outfile:
            for landmarks_frame in landmarks_all:
                num_faces = len(landmarks_frame)
                # print(num_faces)
                if num_faces != 1:
                    file_frame_disorder = "File: " + str(filename) + " Frame: " + str(landmarks_all.index(landmarks_frame)) + " No. of Faces: " + str(num_faces) + "\n"
                    outfile.write(file_frame_disorder)


if __name__ == "__main__":
    main()

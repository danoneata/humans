import os

import numpy as np

import cv2

from constants import LANDMARKS_INDICES


def make_folder(path):
    """Given a path to a file it creates the folder if it is missing."""
    folder, file_ = os.path.split(path)
    os.makedirs(folder, exist_ok=True)


def normalize_face(image, shape):
    """Normalize face to standard position, such that

      (ⅰ) is centered in the image;
      (ⅱ) is rotated such that the eyes lie on a horizontal line (i.e., the
          face is rotated such that the eyes lie along the same y-coordinates).
      (ⅲ) is scaled such that the size of the faces are approximately identical.

    The code is adapted from here:

    https://raw.githubusercontent.com/jrosebr1/imutils/master/imutils/face_utils/facealigner.py
    https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

    """ 
    # desired values
    eye_l_tgt = [0.35, 0.335]
    face_width_tgt = 256
    face_height_tgt = 256

    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = np.stack(shape)

    # extract the left and right eye (x, y)-coordinates
    α1, ω1 = LANDMARKS_INDICES["eye-l"]
    α2, ω2 = LANDMARKS_INDICES["eye-r"]

    eye_l = shape[α1: ω1]
    eye_r = shape[α2: ω2]

    # compute the center of mass for each eye
    center_l = eye_l.mean(axis=0)
    center_r = eye_r.mean(axis=0)

    # compute the angle between the eye centroids
    dX, dY = center_r - center_l
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    eye_r_x_tgt = 1 - eye_l_tgt[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    dist_tgt = (eye_r_x_tgt - eye_l_tgt[0])
    dist_tgt = dist_tgt * face_width_tgt
    scale = dist_tgt / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    center = (center_l + center_r) / 2

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(tuple(center), angle, scale)

    # update the translation component of the matrix
    tX = face_width_tgt * 0.5
    tY = face_height_tgt * eye_l_tgt[0]
    M[0, 2] += (tX - center[0])
    M[1, 2] += (tY - center[1])

    # apply the affine transformation
    w, h = face_width_tgt, face_height_tgt
    output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    # return the aligned face
    return output

import os
import pdb

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
    eye_l_tgt = np.array([0.35, 0.35])
    eye_r_tgt = np.array([0.35, 0.65])
    w_tgt = 256
    h_tgt = 256
    size_tgt = np.array([w_tgt, h_tgt])

    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = np.stack(shape)

    # extract the left and right eye (x, y)-coordinates
    α1, ω1 = LANDMARKS_INDICES["eye-l"]
    α2, ω2 = LANDMARKS_INDICES["eye-r"]

    eye_l = shape[α1: ω1]
    eye_r = shape[α2: ω2]

    # compute the center of mass for each eye
    eye_l_center = eye_l.mean(axis=0)
    eye_r_center = eye_r.mean(axis=0)

    # compute the angle between the eye centroids
    dX, dY = eye_r_center - eye_l_center
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist_src = np.linalg.norm(eye_r_center - eye_l_center)
    dist_tgt = np.linalg.norm(eye_r_tgt - eye_l_tgt) * w_tgt
    scale = dist_tgt / dist_src

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    center = (eye_l_center + eye_r_center) / 2

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(tuple(center), angle, scale)
    # import pdb; pdb.set_trace()

    # update the translation component of the matrix
    tX = w_tgt * 0.5
    tY = h_tgt * eye_l_tgt[0]
    M[0, 2] += (tX - center[0])
    M[1, 2] += (tY - center[1])

    # apply the affine transformation
    output = cv2.warpAffine(image, M, tuple(size_tgt), flags=cv2.INTER_CUBIC)

    def swap(xy):
        x, y = xy
        return y, x
    line_thickness = 2
    p = swap(tuple((eye_l_tgt * size_tgt).astype(int)))
    q = swap(tuple((eye_r_tgt * size_tgt).astype(int)))
    cv2.line(output, p, q, (0, 255, 0), thickness=line_thickness)

    # return the aligned face
    return output


def normalize_face_mouth(image, shape, size, scale=None):
    """Normalize face based on mouth position."""
    # desired values
    l_tgt = np.array([0.5, 0.35])
    r_tgt = np.array([0.5, 0.65])
    size_tgt = np.array(size)
    w_tgt, h_tgt = size
    K = 1.5  # scaling factor

    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = np.stack(shape)

    # lip corners
    l_idxs = [48, 60]
    r_idxs = [54, 64]

    l_center = shape[l_idxs].mean(axis=0)
    r_center = shape[r_idxs].mean(axis=0)

    dX, dY = l_center - r_center
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    if scale is None:
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image

        # extract the left and right eye (x, y)-coordinates
        α1, ω1 = LANDMARKS_INDICES["eye-l"]
        α2, ω2 = LANDMARKS_INDICES["eye-r"]

        eye_l = shape[α1: ω1]
        eye_r = shape[α2: ω2]

        # compute the center of mass for each eye
        eye_l_center = eye_l.mean(axis=0)
        eye_r_center = eye_r.mean(axis=0)

        dist_src = np.linalg.norm(eye_r_center - eye_l_center)
        dist_tgt = np.linalg.norm(r_tgt - l_tgt) * w_tgt * K

        scale = dist_tgt / dist_src

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    center = (l_center + r_center) / 2

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(tuple(center), angle, scale)

    # update the translation component of the matrix
    tX = w_tgt * 0.5
    tY = h_tgt * l_tgt[0]

    M[0, 2] += (tX - center[0])
    M[1, 2] += (tY - center[1])

    # apply the affine transformation
    output = cv2.warpAffine(image, M, tuple(size_tgt), flags=cv2.INTER_CUBIC)

    def swap(xy):
        x, y = xy
        return np.array([y, x])

    # draw line to show lip line
    line_thickness = 2
    p = tuple((swap(l_tgt) * size_tgt).astype(int))
    q = tuple((swap(r_tgt) * size_tgt).astype(int))
    cv2.line(output, p, q, (0, 255, 0), thickness=line_thickness)

    # rotate mouth landmarks
    num_landmarks = len(shape)
    shape1 = np.hstack((shape, np.ones(shape=(num_landmarks, 1))))
    shape1 = shape1 @ M.T

    return output, shape1, scale

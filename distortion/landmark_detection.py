#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

import sys
import os
import dlib
import glob
from skimage import io
from imutils import face_utils
import argparse
import numpy as np
import matplotlib.pyplot as plt

# I modified dlib code. This file save the coordinates of face landmarks from face images.
parser = argparse.ArgumentParser()
parser.add_argument('--part', default = None,  type = str, help = "select a part of face")
parser.add_argument('--dir', type = str, help = 'enter file dir or file name')
args = parser.parse_args()

predictor_path = r'C:\Users\LG\Documents\Deep_learning\caricature\dlib-19.4\shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

if args.dir.strip()[-3:] == 'jpg':
    f = args.dir
    print("Processing file: {}".format(f))
    img = io.imread(f)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))

        # Draw the face landmarks on the screen.

        win.add_overlay(shape)
        coord = face_utils.shape_to_np(shape)

    win.add_overlay(dets)
    dlib.hit_enter_to_continue()

    print("shape : ", coord.shape)
    np.save(os.path.join(os.path.split(args.dir)[0], "coord"), coord)
    print("coords saved!")

else :
    i = 0
    file_list = glob.glob(os.path.join(args.dir, "*.jpg"))
    print('total file number :', len(file_list))

    # file name should be started at 0 and be integer
    coords = [0]*len(file_list)
    box = [0]*len(file_list)
    for f in file_list:
        print("Processing file: {}".format(f))
        img = io.imread(f)

        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        if len(dets) != 1:
            print("file has problem!!", f)
            pass

        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))

            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))

            # Draw the face landmarks on the screen.
            win.add_overlay(shape)

            #change shape type from dlib objective to numpy array
            shape = face_utils.shape_to_np(shape)
            file_num = int(os.path.splitext(os.path.split(f)[1])[0])
            coords[file_num] = shape
            box[file_num] = [d.left(), d.right(), d.top(), d.bottom()]

        win.add_overlay(dets)

        '''
        labels = [x for x in range(68)]
        plt.scatter(coords[i][:, 0], 500 - coords[i][:, 1])
        for label, x, y in zip(labels, coords[i][:, 0], 500 - coords[i][:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(3, 0),
                textcoords='offset points')
        plt.show()
        i += 1
        '''

    dlib.hit_enter_to_continue()

    # save coordinates as numpy file
    coords = np.stack(coords, axis=0)
    box = np.array(box)
    print("coords shape : ",  coords.shape)
    np.save(os.path.join(args.dir, "coords"), coords)
    print("coords saved!")
    np.save(os.path.join(args.dir, "box"), box)
    print("box saved!")

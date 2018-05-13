This folder contains python files related to face distortion and landmark detection


## Files explanation
- landmark_detection 
save the coordinates of face landmarks of face images.  
For landmark detection, I used dlib package. (see landmark_detection.py)

- distort_landmark
Implement distortion of face image using landmark coordinates and distortion function which I made. 

- distort_function
The distortion function which has width, height, curvature as parameter. (We can control the distortion by these parameters.)


## About Dlib face landmark detection
The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt

  This example program shows how to find frontal human faces in an image and
  estimate their pose.  The pose takes the form of 68 landmarks.  These are
  points on the face such as the corners of the mouth, along the eyebrows, on
  the eyes, and so forth.

  This face detector is made using the classic Histogram of Oriented
  Gradients (HOG) feature combined with a linear classifier, an image pyramid,
  and sliding window detection scheme.  The pose estimator was created by
  using dlib's implementation of the paper:
     One Millisecond Face Alignment with an Ensemble of Regression Trees by
     Vahid Kazemi and Josephine Sullivan, CVPR 2014
  and was trained on the iBUG 300-W face landmark dataset.

  Also, note that you can train your own models using dlib's machine learning
  tools. See train_shape_predictor.py to see an example.

  You can get the shape_predictor_68_face_landmarks.dat file from:
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
  You can install dlib using the command:
      pip install dlib

  Alternatively, if you want to compile dlib yourself then go into the dlib
  root folder and run:
      python setup.py install
  or
      python setup.py install --yes USE_AVX_INSTRUCTIONS
  if you have a CPU that supports AVX instructions, since this makes some
  things run faster.  

  Compiling dlib should work on any operating system so long as you have
  CMake and boost-python installed.  On Ubuntu, this can be done easily by
  running the command:
      sudo apt-get install libboost-python-dev cmake

  Also note that this example requires scikit-image which can be installed
  via the command:
      pip install scikit-image
  Or downloaded from http://scikit-image.org/download.html.

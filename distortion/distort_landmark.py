import numpy as np
import argparse
import cv2
from distort_function import distortion
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type = str, help = 'enter file dir or file name')
parser.add_argument('--mag', type = int, default = 1)
args = parser.parse_args()

img = cv2.imread(args.dir)
coords = np.load(os.path.join(os.path.split(args.dir)[0], 'coords.npy'))
coord = coords[int(os.path.splitext(os.path.split(args.dir)[1])[0])]

eye_w = 10; eye_h = 0
nose_w = 50; nose_h = 10
mouse_w = 50; mouse_h = 30
cheek_w = 25; cheek_h = 20
ema_w = 40; ema_h = 40

# coord[i][0] is distance from the left, coord[i][1] is distance from the top

# find center
center_lefteye = (int((np.max(coord[36:42][:,1])+np.min(coord[36:42][:,1]))/2),int((coord[39][0]+coord[36][0])/2))
center_righteye = (int((np.max(coord[42:48][:,1])+np.min(coord[42:48][:,1]))/2),int((coord[45][0]+coord[42][0])/2))
center_nose = (int((np.max(coord[29:35][:,1])+np.min(coord[29:35][:,1]))/2),int((coord[35][0]+coord[31][0])/2))
center_mouse = (int((np.max(coord[48:68][:,1])+np.min(coord[48:68][:,1]))/2),int((coord[54][0]+coord[48][0])/2))
center_leftcheek = (int((coord[2][1]+coord[31][1])/2),int((2*coord[2][0]+coord[31][0])/3))
center_rightcheek = (int((coord[14][1]+coord[35][1])/2),int((2*coord[14][0]+coord[35][0])/3))
center_ema = (int(coord[21][1]+coord[22][1]-coord[29][1]),int((coord[21][0]+coord[22][0])/2))

center = [center_lefteye, center_righteye, center_nose, center_mouse,
          center_leftcheek, center_rightcheek, center_ema]

# find width and height
width_lefteye = int((coord[39][0]-coord[36][0]+1)/2)+eye_w
width_righteye = int((coord[45][0]-coord[42][0]+1)/2)+eye_w
width_nose = int((coord[35][0]-coord[31][0]+1)/2)+nose_w
width_mouse = int((coord[54][0]-coord[48][0]+1)/2)+mouse_w
width_cheek = int((coord[31][0]-coord[2][0]+1)/2)+cheek_w
width_ema = int((coord[16][0]-coord[0][0]+1)/2)+ema_w

width = [width_lefteye, width_righteye, width_nose, width_mouse, width_cheek, width_cheek, width_ema]

# find width and height
height_lefteye = int((np.max(coord[36:42][:,1])-np.min(coord[36:42][:,1])+1)/2)+eye_h
height_righteye = int((np.max(coord[42:48][:,1])-np.min(coord[42:48][:,1])+1)/2)+eye_h
height_nose = int((np.max(coord[29:35][:,1])-np.min(coord[29:35][:,1])+1)/2)+nose_h
height_mouse = int((np.max(coord[48:68][:,1])-np.min(coord[48:68][:,1])+1)/2)+mouse_h
height_cheek = width_cheek - cheek_w + cheek_h
height_ema = int(coord[28][1] - (coord[21][1]+coord[22][1])/2) + ema_h

height = [height_lefteye, height_righteye, height_nose, height_mouse, height_cheek, height_cheek, height_ema]

img_crop = []
for i in range(7):
    img_crop.append(img[center[i][0]-height[i]:center[i][0]+height[i]+1, center[i][1]-width[i]:center[i][1]+width[i]+1])

img_copy = np.copy(img)
for i in range(27,68):
    cv2.circle(img_copy, (coord[i][0], coord[i][1]), 1, (0,255,0), -1)
cv2.circle(img_copy, (center_lefteye[1],center_lefteye[0]), 2, (0,0,255), -1)
cv2.circle(img_copy, (center_righteye[1],center_righteye[0]), 2, (0,0,255), -1)
cv2.circle(img_copy, (center_nose[1],center_nose[0]), 2, (0,0,255), -1)
cv2.circle(img_copy, (center_mouse[1],center_mouse[0]), 2, (0,0,255), -1)
cv2.circle(img_copy, (center_leftcheek[1],center_leftcheek[0]), 2, (0,0,255), -1)
cv2.circle(img_copy, (center_rightcheek[1],center_rightcheek[0]), 2, (0,0,255), -1)
cv2.circle(img_copy, (center_ema[1],center_ema[0]), 2, (0,0,255), -1)


cv2.imshow('img', img_copy)
for i in range(7):
    cv2.imshow('img%d'%i, img_crop[i])

cv2.waitKey(0)
cv2.destroyAllWindows()

# img = distortion(img, 0.0, 3, center[0], width[0], height[0], mode='ver',f='tan', save=False)
# img = distortion(img, 0.0, 3, center[1], width[1], height[1], mode='ver',f='tan', save=False)
# img = distortion(img, 0.0, 4, center[2], width[2], height[2], mode='hor',f='tan', save=False)
# img = distortion(img, 0.0, 3, center[3], width[3], height[3], mode='hor',f='tan',
#                   save=os.path.join(os.path.split(args.dir)[0], 'distorted.jpg'))
#
#
# img = distortion(img, 0.0, 3, center[4], width[4], height[4], mode='hor',f='cos', save=False)
# img = distortion(img, 0.0, 3, center[5], width[5], height[5], mode='hor',f='cos',
#                  save=os.path.join(os.path.split(args.dir)[0], 'distorted.jpg'))
img = distortion(img, 0.0, 3, center[6], width[6], height[6], mode='mix',f='tan',
                 save=os.path.join(os.path.split(args.dir)[0], 'distorted.jpg'))



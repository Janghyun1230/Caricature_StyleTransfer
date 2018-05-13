# make and save relative information

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (5,10)


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type = str, help = 'enter file dir')
args = parser.parse_args()

if not os.path.exists(args.dir) :
    raise Exception("File doesn't exist")

coords = np.load(args.dir).astype(np.float32)
print("coords shape :" ,coords.shape)
print("coords dtype :" ,coords.dtype)
for i, coord in enumerate(np.mean(coords, axis = (1,2))):
    if coord == 0:
        print('%d has error'%i)

# show coordinates
labels = [x for x in range(68)]
plt.subplot(2,1,1)
plt.scatter(coords[0][:,0], 500-coords[0][:,1])
for label, x, y in zip(labels, coords[0][:,0], 500-coords[0][:,1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(3, 0),
        textcoords='offset points')
plt.show()


N = coords.shape[0]
face_info = [['left_eye_width_0', 'right_eye_width_1', 'left_eye_height_2', 'right_eye_height_3', 'nose_width_4', 'nose_height_5',
              'mouse_width_6', 'mouse_height_7', 'face_height_8', 'meegan_9', 'injung_10', 'teeth_11'], []]

for i in range(N) :
    # rescale each image with face width
    width = coords[i][16][0]- coords[i][0][0]
    coords[i] /= width

    # face info
    _0 = coords[i][39][0]- coords[i][36][0]
    _1 = coords[i][45][0] - coords[i][42][0]
    _2 = np.max(coords[i][36:42][:,1]) - np.min(coords[i][36:42][:,1])
    _3 = np.max(coords[i][42:48][:,1]) - np.min(coords[i][42:48][:,1])
    _4 = coords[i][35][0] - coords[i][31][0]
    _5 = -(coords[i][27][1] - coords[i][33][1])
    _6 = coords[i][54][0] - coords[i][48][0]
    _7 = np.max(coords[i][48:68][:,1]) - np.min(coords[i][48:68][:,1])
    _8 = -(coords[i][27][1] - coords[i][8][1])
    _9 = coords[i][42][0] - coords[i][39][0]
    _10 = -(coords[i][33][1] - coords[i][51][1])
    _11 = -(coords[i][62][1] - coords[i][66][1])

    face_info[1].append([_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11])

face_info_agg = np.array(face_info[1])
# add mean
mean = np.mean(face_info_agg, axis = 0)
std = np.std(face_info_agg, axis= 0)
face_info_agg = np.vstack([face_info_agg,mean, std])

print("face_info with mean, std shape : ", face_info_agg.shape)

np.save(os.path.join(os.path.split(args.dir)[0],'face_info'), face_info_agg)
print("face infomation saved!")


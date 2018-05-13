# from relative information draw distribution and find a image

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (5,5)


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type = str, help = 'enter file dir')
parser.add_argument('--info_num', type = int, help = 'enter information number which you want')
parser.add_argument('--img_num', type = int, help = 'enter image number which you want')
args = parser.parse_args()

if not os.path.exists(args.dir) :
    raise Exception("File doesn't exist")

info = np.load(args.dir)
print("numpy size : ", info.shape)
print('mean check', np.mean(info[:-2,], axis = 0)==info[-2])
print('std check', np.std(info[:-2], axis = 0)==info[-1])

info_list = ['left_eye_width_0', 'right_eye_width_1', 'left_eye_height_2', 'right_eye_height_3', 'nose_width_4', 'nose_height_5',
              'mouse_width_6', 'mouse_height_7', 'face_height_8', 'meegan_9', 'injung_10', 'teeth_11']

print(info_list)
print("mean face info : ", info[-2,:])

if args.info_num is not None:
    print(info_list[args.info_num])
    plt.hist(info[:-2,args.info_num], 100)
    plt.title(info_list[args.info_num])
    plt.show()

    print('max image number :', np.argmax(info[:-2,args.info_num]))
    print('min image number :', np.argmin(info[:-2,args.info_num]))


if args.img_num is not None:
    num = args.img_num
    a = (info[num,:] - info[-2,:])/info[-1,:]
    for i in range(12):
        print("%s: %0.2f"%(info_list[i],a[i]))

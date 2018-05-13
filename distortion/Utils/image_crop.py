import numpy as np
import cv2
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type = str, help = 'enter file dir or file name')
parser.add_argument('--size', type = int, help = 'cropped image size')
args = parser.parse_args()

coords = np.load(os.path.join(args.dir,'coords.npy'))
file_list = glob.glob(os.path.join(args.dir, "*.jpg"))

for f in file_list:
    img = cv2.imread(f)
    file_num = int(os.path.splitext(os.path.split(f)[1])[0])
    coord = coords[file_num]
    center = coord[27]
    len = int((coord[8][1] - coord[27][1])*1.3)

    if center[1] - len < 0 :
        len = center[1]
    if center[0] - len < 0:
        len = center[0]
    img_crop = img[center[1]-len:center[1]+len, center[0]-len:center[0]+len]

    if 2*len >= args.size:
        img_crop = cv2.resize(img_crop, (args.size,args.size), interpolation = cv2.INTER_AREA)
    else :
        img_crop = cv2.resize(img_crop, (args.size, args.size), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(os.path.join(args.dir, 'crop\%d.jpg'%file_num), img_crop)


# cv2.imshow("img", img)
# cv2.imshow("img_crop", img_crop)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


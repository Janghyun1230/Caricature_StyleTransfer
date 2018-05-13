import cv2
import glob
import os
from skimage import io

f_jpg = glob.glob(os.path.join(r'C:\Users\LG\Pictures\google_iphoto\new', "*.jpg"))

i=0

for file in f_jpg :
    img = io.imread(file)
    img2 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    path = os.path.join(r'C:\Users\LG\Pictures\google_iphoto\resize1', '%d.jpg'%i)
    i+=1
    io.imsave(path, img2)
    if i%100 == 0 :
        print(i)

print("finish")
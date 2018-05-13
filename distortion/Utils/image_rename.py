import os
import glob
import argparse
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type = str, help = 'enter file dir or file name')
parser.add_argument('--mode', type = str, help = 'enter file dir or file name')
args = parser.parse_args()

f_jpg = glob.glob(os.path.join(args.dir, "*.jpg"))
f_jpeg = glob.glob(os.path.join(args.dir, "*.jpeg"))
f_png = glob.glob(os.path.join(args.dir, "*.png"))
f_gif = glob.glob(os.path.join(args.dir, "*.gif"))
f = f_jpg+f_jpeg+f_png+f_gif

print('total image file len: ', len(f))
print('sample : ', f[0])

i = 0

for file in f_jpg :
    img = io.imread(file)
    path = os.path.join(os.path.split(file)[0], 'new\%d.jpg'%i)
    i+=1
    io.imsave(path, img)
    print(i)

for file in f_jpeg :
    img = io.imread(file)
    path = os.path.join(os.path.split(file)[0], 'new\%d.jpg'%i)
    i+=1
    io.imsave(path, img)
    print(i)

for file in f_png :
    img = io.imread(file)
    path = os.path.join(os.path.split(file)[0], 'new\%d.jpg'%i)
    i+=1
    io.imsave(path, img)
    print(i)

for file in f_gif :
    img = io.imread(file)
    path = os.path.join(os.path.split(file)[0], 'new\%d.jpg'%i)
    i+=1
    io.imsave(path, img)
    print(i)

print("finished!")
import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default = r'C:\Users\LG\Pictures\test_dist.jpg')
parser.add_argument('--hsv', type = bool,  default = True)
parser.add_argument('--painting', type = bool, default = True)
parser.add_argument('--edge', type = str, default = 'sketch')
args = parser.parse_args()

img = cv2.imread(args.dir)
img = cv2.resize(img,(256,256), interpolation=cv2.INTER_AREA)

# hsv
if args.hsv is True:
    term = 13
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_copy = hsv.copy()
    hsv_copy[:, :, 1] = hsv[:, :, 1] // term * term
    img = cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2BGR)

    cv2.imshow('original_img', img)

h,w,_ = img.shape
print(w,h)

# oil painting
if args.painting is True:
    k = 2
    img_copy = np.ones_like(img) * 255
    intensity = 8
    bins = np.array(range(intensity+2))

    for i in range(k,h-k):
        if i%100 == 0:
            print(i)
        for j in range(k,w-k):
            kernal = img[i-k:i+k, j-k:j+k]
            kernal_intensity = np.round(np.sum(kernal, axis = 2)/3.0*intensity/255)
            hist, _ = np.histogram(kernal_intensity, bins = bins)
            most = np.argmax(hist)
            img_copy[i,j] = np.mean(kernal[kernal_intensity==most], axis = 0)
else :
    img_copy = (cv2.bilateralFilter(img,9,75,75))

# draw sketch
if args.edge == 'sketch':
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.add(img_gray,10)

    img_blur = cv2.GaussianBlur(img_gray,(15,15),0)
    img_blend = cv2.divide(img_gray,img_blur, scale = 256)
    cv2.imshow('sketch', img_blend)

    _, mask = cv2.threshold(img_blend, 230, 255, cv2.THRESH_BINARY)
    cv2.imshow('mask', mask)

    img = cv2.bitwise_and(img_copy,img_copy,mask = mask)
    cv2.imshow('image', img)

elif args.edge == 'canny':
    # canny edge
    can = cv2.bitwise_not(cv2.Canny(img,150,200))
    cv2.imshow('canny_edge', can)
    # masking
    img = cv2.bitwise_and(img_copy,img_copy,mask = can)
    cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite(os.path.join(os.path.split(args.dir)[0], "new_image.jpg"), img)




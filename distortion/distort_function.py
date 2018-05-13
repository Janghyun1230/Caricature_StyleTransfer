import numpy as np
import argparse
import cv2


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

# 1. f=tan then set b=0, else b=0.5 will good
# 2. higher b=1 img remains same, lower b=0 stretch widely
# 3. higher c img becomes more concave
def distortion(img, b, c, center, width, height, mode='mix', f='tan', save=False):
    # 1. crop image
    img_crop = img[center[0]-height:center[0]+height+1, center[1]-width:center[1]+width+1]

    # 2. distort
    if mode == 'hor' or mode == 'mix':
        img_dis1 = np.zeros_like(img_crop)
        for i in range(img_crop.shape[0]):
            a = (np.abs(i - height) / np.float(height))**c * (1 - b) + b
            for j in range(img_crop.shape[1]):
                if f == 'arcsin':
                    j_pre = int((1 - a) * np.arcsin((j - width) / width) * 2.0 / np.pi * width + a * (j - width))+width
                elif f == 'cos':
                    j_pre = int((1 - a) * (1.0-np.cos((j - width) / width*np.pi/2.0 ))*sign(j-width) * width + a * (j - width)) + width
                elif f == 'tan':
                    j_pre = int((1 - a) * np.tan((j - width) / width *np.pi/4.0 )* width + a * (j - width)) + width
                else:
                    raise ArgumentError("you should enter right function")
                img_dis1[i, j] = img_crop[i, j_pre]
        img_crop = np.copy(img_dis1)

    if mode == 'ver' or mode == 'mix' :
        img_dis1 = np.zeros_like(img_crop)
        for j in range(img_crop.shape[1]):
            a = (np.abs(j - width) / np.float(width))**c * (1 - b) + b
            for i in range(img_crop.shape[0]):
                if f == 'arcsin':
                    i_pre = int((1 - a) * np.arcsin((i - height) / height) * 2.0 / np.pi * height + a * (i - height)) + height
                elif f == 'cos':
                    i_pre = int((1 - a) * (1.0-np.cos((i - height) / height*np.pi/2.0 ))*sign(i-height)*height + a*(i-height))+height
                elif f == 'tan':
                    i_pre = int((1 - a) * np.tan((i-height)/height*np.pi/4.0 )*height + a*(i-height))+height
                else:
                    raise ArgumentError("you should enter right function")
                img_dis1[i, j] = img_crop[i_pre, j]

    # when a = b array they share the adress ex) a[0] = b[0] same adress
    img_dist = np.copy(img)
    img_dist[center[0]-height:center[0]+height+1, center[1]-width:center[1]+width+1] = img_dis1

    cv2.imshow('girl', img)
    cv2.imshow('girl_dist',img_dist)

    cv2.waitKey(0)

    # by enter can close windows
    cv2.destroyAllWindows()

    if save:
        cv2.imwrite(save,img_dist)

    return img_dist

if __name__ == "__main__" :
    def coordinate(s):
        try :
            x,y = map(int, s.split(','))
            return (x,y)
        except :
            raise argparse.ArgumentTypeError("Coordinates must be x,y")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, default = 'mix', help = 'enter mode ver : vertical, hor : horizontal, mix : both')
    parser.add_argument('--dir', type = str, help = 'enter image directory')
    parser.add_argument('--center', type = coordinate,  help = 'enter center x,y')
    parser.add_argument('--width', type = int, help = 'enter width/2')
    parser.add_argument('--height', type = int, help = 'enter height/2')
    parser.add_argument('-b', type=float, default=0.5, help='enter stretch magnitude [0,1] the lower means the more')
    parser.add_argument('-c', type=float, default=4, help='enter curvature magnitude [0,inf] the larger means the wider')
    parser.add_argument('--save', default = False,  help = 'save or not if you want to save enter directory')
    args = parser.parse_args()

    width = args.width
    height = args.height
    center = args.center
    b = args.b
    c = args.c
    img = cv2.imread(args.dir)

    distortion(img, b, c, center, width, height, args.mode, save=args.save)






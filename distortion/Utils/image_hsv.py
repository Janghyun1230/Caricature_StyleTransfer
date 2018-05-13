import cv2
import numpy as np

img = cv2.imread(r'C:\Users\LG\Documents\Deep learning\style_transfer\test images\painting.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# check! j=1 saturation, j=2 value
j = 1

for i in range(1,20,2):
    term = i
    hsv_copy = hsv.copy()
    hsv_copy[:,:,j] = hsv[:,:,j]//term*term
    img = cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2BGR)
    cv2.imshow('original_img%d'%i, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
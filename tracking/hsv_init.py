import cv2
import numpy as np
import sys

# global image_hsv, pixel # so we can use it in mouse callback

# mouse callback function
def pick_color(event, x, y, flags, param):

    global image_hsv, pixel

    if event == cv2.EVENT_LBUTTONDOWN:

        pixel = image_hsv[y, x]

        # you might want to adjust the ranges(+-10, etc):
        upper = np.array([pixel[0] + 10, pixel[1] + 20, pixel[2] + 40])
        lower = np.array([pixel[0] - 10, pixel[1] - 20, pixel[2] - 40])
        print(pixel, lower, upper)

        mask = cv2.inRange(image_hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # edged = cv2.Canny(fgmask, 30, 100)  # any gradient between 30 and 150 are considered edges
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)  # dilate

        # cv2.imshow("mask", mask)
        # cv2.imshow("mask_dilated", mask_dilated)

        cv2.imshow("mask_dilated", np.hstack([mask, mask_dilated]))




path = 'image/'
# file = 'res.jpg'
file = 'h2.jpeg'

img = cv2.imread(path + file)

img_h, img_w = img.shape[:2]
img_k = 2
img_w = int(img_w/img_k)
img_h = int(img_h/img_k)
img = cv2.resize(img, (img_w, img_h))

cv2.namedWindow('bgr')
cv2.imshow('bgr', img)
cv2.setMouseCallback('bgr', pick_color)

# now click into the hsv img , and look at values:
image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsv", image_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()



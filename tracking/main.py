import cv2
import numpy as np

path = 'input/'


frame = cv2.imread(path + 'h3.jpg')


cv2.imshow('frame', frame)
cv2.waitKey(0)
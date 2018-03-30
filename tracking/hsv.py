import cv2
import numpy as np

def nothing(n):
    pass

# Contours w/ greatest number of points
# TODO max by area
def biggestContourI(contours):
    maxVal = 0
    maxI = None
    for i in range(0, len(contours) - 1):
        if len(contours[i]) > maxVal:
            cs = contours[i]
            maxVal = len(contours[i])
            maxI = i
    return maxI


path = 'image/'
file = 'h2.jpeg'

img = cv2.imread(path + file)

img_h, img_w = img.shape[:2]
img_k = 2
img_w = int(img_w/img_k)
img_h = int(img_h/img_k)
img = cv2.resize(img, (img_w, img_h))

# iLowH = 155
# iHighH = 225
# iLowS = 47
# iHighS = 126
# iLowV = 82
# iHighV = 132

iLowH = 0
iHighH = 179
iLowS = 0
iHighS = 255
iLowV = 0
iHighV = 255

hsv_ranges = {
    1: [[110, 50, 50], [130, 255, 255]],  # blue
    2: [[0, 0, 0], [15, 255, 255]],  # red
}

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

cv2.namedWindow('Control', cv2.WINDOW_NORMAL)
cv2.createTrackbar("LowH", "Control", iLowH, 179, nothing)
cv2.createTrackbar("HighH", "Control", iHighH, 179, nothing)
cv2.createTrackbar("LowS", "Control", iLowS, 255, nothing)
cv2.createTrackbar("HighS", "Control", iHighS, 255, nothing)
cv2.createTrackbar("LowV", "Control", iLowV, 255, nothing)
cv2.createTrackbar("HighV", "Control", iHighV, 255, nothing)

switch = '0 : None\n1 : Blue\n2 : Red'
cv2.createTrackbar(switch, "Control", 0, 2, nothing)

cv2.createTrackbar('Correct', "Control", 0, 1, nothing)
cv2.setTrackbarPos('Correct', 'Control', 1)

cam = cv2.VideoCapture(0)


while True:

    lh = cv2.getTrackbarPos('LowH', 'Control')
    ls = cv2.getTrackbarPos('LowS', 'Control')
    lv = cv2.getTrackbarPos('LowV', 'Control')
    hh = cv2.getTrackbarPos('HighH', 'Control')
    hs = cv2.getTrackbarPos('HighS', 'Control')
    hv = cv2.getTrackbarPos('HighV', 'Control')
    c = cv2.getTrackbarPos(switch, 'Control')
    correct = cv2.getTrackbarPos('Correct', 'Control')

    if c != 0 and correct != 1:
        cv2.setTrackbarPos('HighH', 'Control', hsv_ranges[c][1][0])
        cv2.setTrackbarPos('HighS', 'Control', hsv_ranges[c][1][1])
        cv2.setTrackbarPos('HighV', 'Control', hsv_ranges[c][1][2])
        cv2.setTrackbarPos('LowH', 'Control', hsv_ranges[c][0][0])
        cv2.setTrackbarPos('LowS', 'Control', hsv_ranges[c][0][1])
        cv2.setTrackbarPos('LowV', 'Control', hsv_ranges[c][0][2])
        # cv2.setTrackbarPos('Correct', 'Control', 0)

    if c == 0 and correct != 1:
        cv2.setTrackbarPos('HighH', 'Control', iHighH)
        cv2.setTrackbarPos('HighS', 'Control', iHighS)
        cv2.setTrackbarPos('HighV', 'Control', iHighV)
        cv2.setTrackbarPos('LowH', 'Control', iLowH)
        cv2.setTrackbarPos('LowS', 'Control', iLowS)
        cv2.setTrackbarPos('LowV', 'Control', iLowV)
        # cv2.setTrackbarPos('Correct', 'Control', 0)


    lower = np.array([lh, ls, lv], dtype="uint8")
    higher = np.array([hh, hs, hv], dtype="uint8")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    mask = cv2.inRange(hsv, lower, higher)

    im2, contours0, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # Only draw the biggest one
    # bc = biggestContourI(contours0)
    # cv2.drawContours(img, contours0, bc, (0, 255, 0), 3)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    #
    # cv2.imshow('my webcam', img)

    # cv2.imshow('img', img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)

    cv2.imshow('res', np.hstack([img, res]))

    if cv2.waitKey(1) == 27:
        break  # esc to quit

cv2.destroyAllWindows()



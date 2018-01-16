import cv2
import numpy as np

path = 'input/'


frame = cv2.imread(path + 'h1.jpg')

pts1 = []
pts2 = []

im_disp = frame.copy()
im_draw = frame.copy()

# window_name = "Select objects to be tracked here."
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.imshow(window_name, im_draw)
#
# mouse_down = False
#
# def callback(event, x, y, flags, param):
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(pts1) == 1:
#             print("WARN: Cannot select another object in SINGLE OBJECT TRACKING MODE.")
#             print("Delete the previously selected object using key `d` to mark a new location.")
#             return
#         mouse_down = True
#         pts1.append((x, y))
#
#     elif event == cv2.EVENT_LBUTTONUP and mouse_down == True:
#         mouse_down = False
#         pts2.append((x, y))
#         print("Object selected at [{}, {}]".format(pts1[-1], pts2[-1]))
#
#     elif event == cv2.EVENT_MOUSEMOVE and mouse_down == True:
#         im_draw = frame.copy()
#         cv2.rectangle(im_draw, pts1[-1], (x, y), (255, 255, 255), 3)
#         cv2.imshow(window_name, im_draw)
#
#
# cv2.setMouseCallback(window_name, callback)
#
# print(pts1)
# print(pts2)
#
# # cv2.imshow('frame', frame)
# cv2.waitKey(0)


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def callback(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        cv2.line(im_draw, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", im_draw)


cv2.namedWindow("image")
cv2.setMouseCallback("image", callback)

if len(refPt) == 2:
    # draw a rectangle around the region of interest
    cv2.line(im_draw, refPt[0], refPt[1], (0, 0, 255), 2)
    cv2.imshow("image", im_draw)

cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()


# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

# https://github.com/bikz05/object-tracker/blob/master/get_points.py

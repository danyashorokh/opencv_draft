import cv2
import numpy as np

path = 'input/'


frame = cv2.imread(path + 'h1.jpg')
field = cv2.imread(path + 'field2.png')

pts2 = [(220,5),
        (220, 110), # 216
        (33, 97),
        (33, 123)
        ]

# def callback2(event, x, y, flags, param):
#
#     if event == cv2.EVENT_MOUSEMOVE:
#         print(x,y)
#
# cv2.namedWindow('field')
# # cv2.setMouseCallback('field', callback2)
#
# for p in pts2:
#     cv2.circle(field, p, 5, (0, 0, 255), 2)
#
# cv2.imshow('field', field)
# cv2.waitKey(0)
#
# exit()
####################################

k = 4
frame = cv2.resize(frame, (int(frame.shape[1] / k), int(frame.shape[0] / k)))


im_disp = frame.copy()
im_draw = frame.copy()
field1 = field.copy()

pts = []
ppts = (0,0,1)


def do_nothing(event, x, y, flags, param):
    pass


def set_field(event, x, y, flags, param):
    # grab references to the global variables
    global pts

    if event == cv2.EVENT_LBUTTONDOWN:
        # print('mouse down: x = %s y = %s' % (x, y))

        if len(pts) >= 4:
            pts = []

    elif event == cv2.EVENT_LBUTTONUP:

        # print('mouse up: x = %s y = %s' % (x, y))
        pts.append((x, y))
        if len(pts) < 3:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.circle(im_draw, (x, y), 5, color, 2)
        # cv2.imshow("image", im_draw)

        print(pts)

def set_player(event, x, y, flags, param):
    # grab references to the global variables
    global ppts

    if event == cv2.EVENT_LBUTTONUP:

        print('player x = %s y = %s' % (x,y))

        ppts = (x, y, 1)

        print(ppts)

        hppt = list(np.dot(h, ppts))
        print(hppt)

        hppt[0] /= hppt[2]
        hppt[1] /= hppt[2]

        print(hppt[0], hppt[1])

        cv2.circle(field1, (int(hppt[0]), int(hppt[1])), 7, (0,0,0), 2)

    # elif event == cv2.EVENT_MOUSEMOVE:
    #     print(x,y)


cv2.namedWindow("image")
cv2.imshow("image", im_draw)
# cv2.setMouseCallback("image", set_field)



while(1):

    cv2.setMouseCallback("image", set_field)
    cv2.imshow('image', im_draw)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

    # if the 'r' key is pressed, reset the field points
    if k in [ord("r"), ord("к")]:
        im_draw = frame.copy()
        pts = []

    if len(pts) == 4:
        break

cv2.setMouseCallback("image", do_nothing)
print(pts)






if len(pts) == 4:
    # draw a rectangle around the region of interest
    cv2.line(im_draw, pts[0], pts[1], (0, 255, 0), 2)
    cv2.line(im_draw, pts[2], pts[3], (255, 0, 0), 2)
    cv2.imshow("image", im_draw)


# Do homography
h, status = cv2.findHomography(np.array(pts), np.array(pts2))

# print(h)

np.savetxt('h_matrix.txt', h)
h = np.loadtxt('h_matrix.txt')

# Warp source image to destination based on homography
frame1 = cv2.warpPerspective(frame, h, (field.shape[1], field.shape[0]))


# Display images
# cv2.imshow("Source Image", im_src)
# cv2.imshow("Destination Image", field)
cv2.imshow("Warped Source Image", frame1)

# print(frame1.shape)
# print(field.shape)

# Draw players
while(1):

    cv2.setMouseCallback("image", set_player)
    cv2.imshow('image', im_draw)
    cv2.imshow('field', field1)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

    # if the 'r' key is pressed, reset the field points
    if k in [ord("r"), ord("к")]:
        im_draw = field1.copy()


cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()


# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

# https://github.com/bikz05/object-tracker/blob/master/get_points.py

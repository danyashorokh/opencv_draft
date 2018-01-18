

import sys
import cv2
import numpy as np
from collections import deque

path = 'input/'


frame = cv2.imread(path + 'h1.jpg')
field = cv2.imread(path + 'field2.png')

pts2 = [(272,62),
        (272, 162), # 216-c_down 110-c_center
        (407, 12), # 97-g_top
        (407, 125)
        ]

def callback2(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE:
        print(x,y)
#
# cv2.namedWindow('field')
# cv2.setMouseCallback('field', callback2)
#
# for p in pts2:
#     cv2.circle(field, p, 5, (0, 0, 255), 2)
#
# cv2.imshow('field', field)
# cv2.waitKey(0)
#
# exit()
####################################



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

        # print(pts)

def set_player(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        pass

    elif event == cv2.EVENT_LBUTTONUP:

        hx, hy = h_points(h, x, y)

        print('player x = %s y = %s hx = %s hy = %s' % (x, y, hx, hy))

        # print(hppt[0], hppt[1])

        cv2.circle(field1, (hx, hy), 7, (0,0,0), 2)
        cv2.circle(im_draw1, (x, y), 4, (0,0,0), 2)

    # elif event == cv2.EVENT_MOUSEMOVE:
    #     print(x,y)

def h_points(h, x, y):

    hppt = list(np.dot(h, (x, y, 1)))

    hppt[0] /= hppt[2]
    hppt[1] /= hppt[2]

    hx = int(hppt[0])
    hy = int(hppt[1])

    return hx, hy

k = 2
# frame = cv2.resize(frame, (int(frame.shape[1] / k), int(frame.shape[0] / k)))



#         tracker = cv2.TrackerBoosting_create()
#         tracker = cv2.TrackerMIL_create()
#         tracker = cv2.TrackerKCF_create()
#         tracker = cv2.TrackerTLD_create()
#         tracker = cv2.TrackerMedianFlow_create()
#         tracker = cv2.TrackerGOTURN_create()

tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerMedianFlow_create()
tracker_type = 'MIL'
# tracker = cv2.TRACKER_KCF_CN


path = "videos/area1.mov"


video = cv2.VideoCapture(path)



# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()


video_h, video_w = frame.shape[:2]

video_k = 2
video_w = int(video_w/video_k)
video_h = int(video_h/video_k)

frame = cv2.resize(frame, (video_w, video_h))


im_draw = frame.copy()
im_draw1 = im_draw.copy()
field1 = field.copy()


pts = []


cv2.namedWindow("image")
cv2.imshow("image", im_draw)
# cv2.setMouseCallback("image", set_field)

# pts = [(272, 129), (286, 168), (70, 158), (62, 167)]  # Костыль
# pts = [(542, 260), (588, 374), (198, 264), (122, 336)]

# pts = [(122, 280), (28, 499), (729, 195), (912, 324)]  # full video
pts = [(60, 141), (14, 248), (365, 97), (456, 161)]  # half video



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
    cv2.line(im_draw1, pts[0], pts[1], (0, 255, 0), 2)
    cv2.line(im_draw1, pts[2], pts[3], (255, 0, 0), 2)
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

# cv2.destroyAllWindows()
# cv2.namedWindow("image")
# cv2.namedWindow("field")
# cv2.imshow('image', im_draw)
# cv2.imshow('field', field1)

# cv2.setMouseCallback("image", set_player)
#
# # Draw players
# while(1):
#
#
#     cv2.imshow('image', im_draw1)
#     cv2.imshow('field', field1)
#
#     k = cv2.waitKey(20) & 0xFF
#     if k == 27:
#         break
#
#     # if the 'r' key is pressed, reset the field points
#     if k in [ord("r"), ord("к")]:
#         field1 = field.copy()
#         im_draw1 = im_draw.copy()




# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

# https://github.com/bikz05/object-tracker/blob/master/get_points.py


#         tracker = cv2.TrackerBoosting_create()
#         tracker = cv2.TrackerMIL_create()
#         tracker = cv2.TrackerKCF_create()
#         tracker = cv2.TrackerTLD_create()
#         tracker = cv2.TrackerMedianFlow_create()
#         tracker = cv2.TrackerGOTURN_create()

# tracker = cv2.TrackerMIL_create()
tracker = cv2.TrackerMedianFlow_create()
tracker_type = 'MIL'
# tracker = cv2.TRACKER_KCF_CN


path = "videos/area1.mov"


video = cv2.VideoCapture(path)



# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()






if not ok:
    print('Cannot read video file')
    sys.exit()

frame = cv2.resize(frame, (video_w, video_h))
# frame = cv2.resize(frame, (640, 360))

# Define an initial bounding box
# bbox = (287, 23, 86, 320)

cv2.destroyWindow("image")

# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

buffer = 20
draw_pts = deque(maxlen=buffer)

counter = 0

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    frame = cv2.resize(frame, (video_w, video_h))

    # Update tracker
    ok, bbox = tracker.update(frame)

    bbox_point = (int(bbox[0]) + int(int(bbox[2]) / 2), int(bbox[1] + bbox[3]))

    hx, hy = h_points(h, bbox_point[0], bbox_point[1])

    draw_pts.appendleft((hx, hy))

    # print(hx, hy)
    # field1 = field.copy()
    # cv2.circle(field1, (hx, hy), 9, (0, 0, 255), 2)


    #########

    # loop over the set of tracked points
    for i in np.arange(1, len(draw_pts)):
        # if either of the tracked points are None, ignore
        # them
        if draw_pts[i - 1] is None or draw_pts[i] is None:
            continue

        # check to see if enough points have been accumulated in
        # the buffer
        if counter >= 10 and i == 1 and draw_pts[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            # dX = draw_pts[-10][0] - draw_pts[i][0]
            # dY = draw_pts[-10][1] - draw_pts[i][1]
            # (dirX, dirY) = ("", "")

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
            cv2.line(field1, draw_pts[i - 1], draw_pts[i], (0, 0, 255), thickness)



    #########

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255 ,0 ,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100 ,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75 ,(0 ,0 ,255) ,2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100 ,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50 ,170 ,50) ,2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100 ,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50 ,170 ,50), 2);

    # Display result
    # cv2.imshow("Tracking", frame)
    # cv2.imshow('field', field1)

    # field2 = field1.copy()
    # field_k = frame.shape[0]/field2.shape[0]
    field_k = field1.shape[0]/frame.shape[0]
    # field2 = cv2.resize(field2, (int(field_k * field2.shape[1]), frame.shape[0]))
    frame = cv2.resize(frame, (int(field_k * frame.shape[1]), field1.shape[0]))

    total = np.hstack([field1, frame])
    cv2.imshow("Tracking", total)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break

    counter += 1


cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()
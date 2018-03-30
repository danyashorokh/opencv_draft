
import sys
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import init_points as ip

def print_coords(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE:
        print(x,y)

def do_nothing(event, x, y, flags, param):
    pass

def set_field_points(event, x, y, flags, param):
    # grab references to the global variables
    global field_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # print('mouse down: x = %s y = %s' % (x, y))

        if len(field_points) >= 4:
            field_points = []

    elif event == cv2.EVENT_LBUTTONUP:

        # print('mouse up: x = %s y = %s' % (x, y))
        field_points.append((x, y))
        if len(field_points) < 3:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.circle(field_draw, (x, y), 5, color, 2)

def set_frame_points(event, x, y, flags, param):
    # grab references to the global variables
    global frame_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # print('mouse down: x = %s y = %s' % (x, y))

        if len(frame_points) >= 4:
            frame_points = []

    elif event == cv2.EVENT_LBUTTONUP:

        # print('mouse up: x = %s y = %s' % (x, y))
        frame_points.append((x, y))
        if len(frame_points) < 3:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.circle(frame_draw, (x, y), 5, color, 2)


def set_fborders(event, x, y, flags, param):
    # grab references to the global variables
    global fborders

    if event == cv2.EVENT_LBUTTONUP:

        # print('mouse up: x = %s y = %s' % (x, y))
        fborders.append((x, y))
        cv2.circle(frame_draw, (x, y), 6, (0,0,255), 2)

def set_player(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        pass

    elif event == cv2.EVENT_LBUTTONUP:

        hx, hy = h_points(h, x, y)

        # print('player x = %s y = %s hx = %s hy = %s' % (x, y, hx, hy))

        cv2.circle(field1, (hx, hy), 7, (0,0,0), 2)
        cv2.circle(frame_draw1, (x, y), 4, (0,0,0), 2)

def h_points(h, x, y):

    hppt = list(np.dot(h, (x, y, 1)))

    hppt[0] /= hppt[2]
    hppt[1] /= hppt[2]

    hx = int(hppt[0])
    hy = int(hppt[1])

    return hx, hy

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


def filter_by_hist(field, obj_hist):

    field_hsv = cv2.cvtColor(field.copy(), cv2.COLOR_BGR2HSV)

    obj_by_hist = cv2.calcBackProject([field_hsv], [0, 1], obj_hist, [0, 180, 0, 256], 1)

    # # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(obj_by_hist, -1, disc, obj_by_hist)

    ret, thresh = cv2.threshold(obj_by_hist, 50, 255, 0)
    thresh = cv2.merge((thresh, thresh, thresh))

    team_mask = cv2.bitwise_and(field, thresh)

    return team_mask


# mouse callback function
def pick_color(event, x, y, flags, param):

    global image_hsv, pixel, hsv_ranges

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(hsv_ranges) >= 2:
            hsv_ranges = []

        pixel = image_hsv[y, x]

        # you might want to adjust the ranges(+-10, etc):
        upper = np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower = np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])

        for i in range(len(lower)):
            if lower[i] < 0: lower[i] = 0
            if upper[i] < 0: upper[i] = 0

        print(pixel, lower, upper)

        hsv_ranges.append([lower, upper])

        image_mask = cv2.inRange(image_hsv, lower, upper)
        cv2.imshow("mask", image_mask)

def filter_by_hsv(img, hsv_ranges):

    mask_res = np.ones((img.shape[0], img.shape[1], 1), np.uint8)
    masks = []
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for hsv_range in hsv_ranges:
        masks.append(cv2.inRange(image_hsv, hsv_range[0], hsv_range[1]))

    for mask in masks:
        mask_res = cv2.bitwise_or(mask_res, mask)

    cv2.imshow('mask_res', mask_res)
    cv2.waitKey(0)

    res = cv2.bitwise_and(img, img, mask=mask_res)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    exit()

    return res




# ----------------- MAIN ----------------------------

path_field = 'image/'
path_video = 'video/'

video_file = 'area1.mov'
# video_file = 'wide1.mp4'

# frame = cv2.imread(path + 'h1.jpg')
field = cv2.imread(path_field + 'field2.png')


field_points = []
frame_points = []
fborders = []
hsv_team_1 = []
hsv_team_2 = []

video_k = 2

# ------------------ INIT -----------------------

field_points = ip.field_points
frame_points = ip.frame_points
fborders = ip.fborders

# fborders = [(1, 95), (99, 90), (214, 82), (313, 78), (386, 83), (437, 94), (477, 110),
#             (532, 141), (584, 168), (628, 214), (603, 263), (506, 290), (380, 310), (216, 332), (0, 343)]


# --------------------------------------

# tracker = cv2.TrackerBoosting_create()
# tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerKCF_create()
# tracker = cv2.TrackerTLD_create()
# tracker = cv2.TrackerMedianFlow_create()
# tracker = cv2.TrackerGOTURN_create()

# tracker = cv2.TrackerTLD_create()
tracker = cv2.TrackerMedianFlow_create()
tracker_type = 'MIL'

# --------------------------------------

field_draw = field.copy()

cv2.namedWindow("field")

if not field_points:
    print("Set field points. Press 'r' to reset points. When you chose all points press 's'")

# set field point
if not field_points:

    while(1):

        cv2.setMouseCallback("field", set_field_points)
        cv2.imshow('field', field_draw)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

        # if the 'r' key is pressed, reset the field points
        if k in [ord("r"), ord("к")]:
            field_draw = field.copy()
            field_points = []

        if len(field_points) == 4 and k == ord("s"):
            print("Fields points are chosen:", field_points)
            break

        if len(field_points) < 4 and k == ord("s"):
            print("You must choose more field points. You chose only %s points" % len(field_points))

else:
    print("Field points are already initialized: ", field_points)

for p in field_points:
    cv2.circle(field_draw, p, 5, (0, 0, 255), 2)

cv2.line(field_draw, field_points[0], field_points[1], (0, 0, 255), 2)
cv2.line(field_draw, field_points[1], field_points[3], (0, 0, 255), 2)
cv2.line(field_draw, field_points[2], field_points[3], (0, 0, 255), 2)
cv2.line(field_draw, field_points[2], field_points[0], (0, 0, 255), 2)

# cv2.polylines(field, [np.array(field_points)], True, (0, 0, 255))


cv2.imshow('field', field_draw)
print("Press any key to continue")
# cv2.waitKey(0)
cv2.destroyWindow("field")

video = cv2.VideoCapture(path_video + video_file)

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
video_w = int(video_w/video_k)
video_h = int(video_h/video_k)

frame = cv2.resize(frame, (video_w, video_h))

frame_draw = frame.copy()
frame_draw1 = frame_draw.copy()
field1 = field.copy()

cv2.namedWindow("frame")
cv2.imshow("frame", frame_draw)

# set frame points
if not frame_points:

    print("Set frame points. Press 'r' to reset points. When you chose all points press 's'")

    while(1):

        cv2.setMouseCallback("frame", set_frame_points)
        cv2.imshow('frame', frame_draw)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

        # if the 'r' key is pressed, reset the field points
        if k in [ord("r"), ord("к")]:
            frame_draw = frame.copy()
            frame_points = []

        if len(frame_points) == 4 and k == ord("s"):
            print("Frame points are chosen:", frame_points)
            break

        if len(frame_points) < 4 and k == ord("s"):
            print("You must choose more frame points. You chose only %s points" % len(frame_points))

else:
    print("Frame points are already initialized: ", frame_points)

for p in frame_points:
    cv2.circle(frame_draw, p, 5, (0, 0, 255), 2)

cv2.line(frame_draw, frame_points[0], frame_points[1], (0, 0, 255), 2)
cv2.line(frame_draw, frame_points[1], frame_points[3], (0, 0, 255), 2)
cv2.line(frame_draw, frame_points[2], frame_points[3], (0, 0, 255), 2)
cv2.line(frame_draw, frame_points[2], frame_points[0], (0, 0, 255), 2)

cv2.setMouseCallback("frame", do_nothing)

cv2.imshow('frame', frame_draw)
print("Press any key to continue")
# cv2.waitKey(0)
cv2.destroyWindow("frame")

# Do homography
h, status = cv2.findHomography(np.array(frame_points), np.array(field_points))

# print(h)

np.savetxt('h_matrix.txt', h)
h = np.loadtxt('h_matrix.txt')

# Warp source image to destination based on homography
frame1 = cv2.warpPerspective(frame, h, (field.shape[1], field.shape[0]))

# cv2.imshow("warped", frame1)
# print("Press any key to continue")
# cv2.waitKey(0)
# cv2.destroyWindow("warped")

frame_draw = frame.copy()
cv2.namedWindow("frame")

# set frame points
if not fborders:

    print("Select field borders. Press 'r' to reset borders. When you chose all borders press 's'")

    cv2.setMouseCallback("frame", set_fborders)

    while(1):
        cv2.imshow('frame', frame_draw)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

        # if the 'r' key is pressed, reset the field points
        if k in [ord("r"), ord("к")]:
            frame_draw = frame.copy()
            fborders = []

        if k == ord("s"):
            print("Frame points are chosen:", fborders)
            break
else:
    print("Field borders points are already initialized: ", fborders)



cv2.setMouseCallback("frame", do_nothing)
cv2.polylines(frame_draw, [np.array(fborders)], 1, (0,0,255))

field_mask = np.zeros(frame.shape[:2], dtype="uint8")

cv2.fillPoly(field_mask,[np.array(fborders)], 255, 1)

frame_draw = cv2.bitwise_and(frame_draw, frame_draw, mask=field_mask)

cv2.imshow('frame', frame_draw)

# print("Press any key to continue")
# cv2.waitKey(0)
# cv2.destroyWindow("frame")


frame = cv2.bitwise_and(frame, frame, mask=field_mask)

masked_field = cv2.bitwise_and(frame, frame, mask=field_mask)

# cv2.imwrite('image/res.jpg', masked_field)
# exit()


image_hsv = cv2.cvtColor(masked_field, cv2.COLOR_BGR2HSV)
hsv_ranges = []
#  ------------- team 1 hsv -----------
# team1_init_box = (149, 132, 47, 48)

if len(hsv_team_1) < 2:

    cv2.namedWindow('hsv1')
    cv2.imshow('hsv1', masked_field)
    cv2.setMouseCallback('hsv1', pick_color)

    print("Select two colors for Team 1. Press 'r' to re-select colors. Press 's' to finish")

    while(1):

        # cv2.imshow('frame', frame_draw)

        hsv_team_1 = hsv_ranges

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

        # if the 'r' key is pressed, reset the field points
        if k in [ord("r"), ord("к")]:
            hsv_team_1 = []

        if k == ord("s"):
            if len(hsv_team_1) == 2:
                print("Color for Team 1")
                print(hsv_team_1)
                break
            else:
                print("You must chose 2 colors for Team 1")



else:
    print("Color for Team 1 are already initialized")

hsv_team_1 = filter_by_hsv(masked_field, hsv_team_1)

cv2.imshow('hsv_team_1', hsv_team_1)
cv2.waitKey(0)



exit()

#  ------------- team 2 -----------




# print("Select any object")

# Select a bounding box
bbox = (161, 141, 7, 13)
# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
track_window = bbox
# ------- histogram end -----------

if bbox:
    print("Object is chosen")

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

buffer = 50
draw_pts = deque(maxlen=buffer)

counter = 0

fgbg1 = cv2.createBackgroundSubtractorMOG2()
fgbg2 = cv2.createBackgroundSubtractorKNN()

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

w1 = 4
sc1 = 1.5
pad1 = 8

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()
    frame = cv2.resize(frame, (video_w, video_h))

    frame2 = frame.copy()

    # -------- meanshift -------------

    hsv_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
    mean_dst = cv2.calcBackProject([hsv_frame], [0], roi_hist, [0, 180], 1)

    cv2.imshow('mean_dst', mean_dst)

    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(mean_dst, track_window, term_crit)

    # Draw it on image
    xm, ym, wm, hm = track_window
    cv2.rectangle(frame, (xm, ym), (xm + wm, ym + hm), (0,255,255), 2, 1)

    # -------- meanshift end ---------

    # -------- fgbg ------------------
    frame2 = cv2.GaussianBlur(frame2, (3, 3), 0)

    fgmask = fgbg1.apply(frame2.copy())
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    fgmask = cv2.bitwise_and(fgmask, fgmask, mask=field_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edged = cv2.Canny(fgmask, 30, 100)  # any gradient between 30 and 150 are considered edges
    edged = cv2.dilate(edged, kernel, iterations=2)


    # kernel = np.ones((5, 5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow("background", np.vstack([edged, fgmask]))

    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(cnts):
        # if the contour is too small, ignore it

        M = cv2.moments(cnts[i])
        if M['m00'] == 0:
            pass
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        if cv2.contourArea(c) > 200: #and cx < 150 and cy > 150:

            (x1, y1, w1, h1) = cv2.boundingRect(c)
            # if h1 >= w1:
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

    # ------------   fgbg end --------

    # ------------ hog ------------

    # found, w = hog.detectMultiScale(frame, winStride=(w1, w1), scale=sc1)
    # found_filtered = []
    # for ri, r in enumerate(found):
    #     for qi, q in enumerate(found):
    #         if ri != qi and inside(r, q):
    #             break
    #     else:
    #         found_filtered.append(r)
    #
    # draw_detections(frame, found)
    # draw_detections(frame, found_filtered, 3)
    # ----------- hog end ---------

    # Update tracker
    ok, bbox = tracker.update(frame)

    bbox_point = (int(bbox[0]) + int(int(bbox[2]) / 2), int(bbox[1] + bbox[3]))

    hx, hy = h_points(h, bbox_point[0], bbox_point[1])

    draw_pts.appendleft((hx, hy))

    # print(hx, hy)
    field1 = field.copy()
    cv2.circle(field1, (hx, hy), 9, (0, 0, 255), 2)


    # loop over the set of tracked points
    for i in np.arange(1, len(draw_pts)):
        # if either of the tracked points are None, ignore
        # them
        if draw_pts[i - 1] is None or draw_pts[i] is None:
            continue

        thickness = int(np.sqrt(25 / float(i + 1)) * 2.5)
        cv2.line(field1, draw_pts[i - 1], draw_pts[i], (0, 0, 255), thickness)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75 ,(0 ,0 ,255) ,2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50 ,170 ,50) ,2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50 ,170 ,50), 2)
    cv2.putText(frame, "w1 : " + str((w1)), (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50 ,170 ,50), 2)
    cv2.putText(frame, "sc1 : " + str((sc1)), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50 ,170 ,50), 2)
    cv2.putText(frame, "pad1 : " + str((pad1)), (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50 ,170 ,50), 2)

    # Display result
    # cv2.imshow("Tracking", frame)
    # cv2.imshow('field', field1)

    field_k = field1.shape[0]/frame.shape[0]
    frame = cv2.resize(frame, (int(field_k * frame.shape[1]), field1.shape[0]))

    total = np.hstack([field1, frame])
    cv2.imshow("Tracking", total)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break

    counter += 1

    if k == ord("q"): w1 += 4
    if k == ord("a"): w1 -= 4
    if k == ord("w"): sc1 += 0.01
    if k == ord("s"): sc1 -= 0.01
    if k == ord("e"): pad1 *= 2
    if k == ord("d"): pad1 /= 2


cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()


# ttps://www.youtube.com/watch?v=KOsgEsY8UWI

# https://www.pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/

# https://docs.opencv.org/3.0-beta/modules/tracking/doc/tracking.html

# https://github.com/dev-labs-bg/football-stats
# https://github.com/AndresGalaviz/Football-Player-Tracking
# https://github.com/dhingratul/Player-Tracking/blob/master/tracker_OTS.py

# http://savvastjortjoglou.com/nba-play-by-play-movements.html

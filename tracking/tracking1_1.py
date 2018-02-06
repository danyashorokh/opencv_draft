import sys
import cv2
import numpy as np
from collections import deque

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


path_field = 'image/'
path_video = 'video/'

video_file = 'area1.mov'
# video_file = 'wide1.mp4'

# frame = cv2.imread(path + 'h1.jpg')
field = cv2.imread(path_field + 'field2.png')

field_points = []
frame_points = []

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

field_points = [(272,62),
        (272, 162), # 216-c_down 110-c_center
        (407, 12), # 97-g_top
        (407, 125)
        ]

frame_points = [(60, 141), (14, 248), (365, 97), (456, 161)]  # half video


# frame_points = [(122, 280), (28, 499), (729, 195), (912, 324)]  # full video

# --------------------------------------

field_draw = field.copy()

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
video_k = 2
video_w = int(video_w/video_k)
video_h = int(video_h/video_k)

frame = cv2.resize(frame, (video_w, video_h))

frame_draw = frame.copy()
frame_draw1 = frame_draw.copy()
field1 = field.copy()

cv2.namedWindow("frame")
cv2.imshow("frame", frame_draw)

if not frame_points:
    print("Set frame points. Press 'r' to reset points. When you chose all points press 's'")

# set frame points
if not frame_points:

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

print("Select any object")

# Select a bounding box
bbox = (161, 141, 7, 13)
# bbox = cv2.selectROI('roi', frame, False)
# print(bbox)
# cv2.destroyWindow("roi")

if bbox:
    print("Object is chosen")

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

buffer = 50
draw_pts = deque(maxlen=buffer)

counter = 0

fgbg1 = cv2.createBackgroundSubtractorMOG2()
fgbg2 = cv2.createBackgroundSubtractorKNN()


while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()
    frame = cv2.resize(frame, (video_w, video_h))

    frame2 = frame.copy()

    frame2 = cv2.GaussianBlur(frame2, (3, 3), 0)

    fgmask = fgbg2.apply(frame2.copy())

    kernel = np.ones((3, 3), np.uint8)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("background", fgmask)


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
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
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

    field_k = field1.shape[0]/frame.shape[0]
    frame = cv2.resize(frame, (int(field_k * frame.shape[1]), field1.shape[0]))

    total = np.hstack([field1, frame])
    # cv2.imshow("Tracking", total)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break

    counter += 1


cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()


# ttps://www.youtube.com/watch?v=KOsgEsY8UWI
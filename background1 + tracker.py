
import numpy as np
import cv2
import init_points as ip

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


class BackGroundSubtractor:
    # When constructing background subtractor, we
    # take in two arguments:
    # 1) alpha: The background learning factor, its value should
    # be between 0 and 1. The higher the value, the more quickly
    # your program learns the changes in the background. Therefore,
    # for a static background use a lower value, like 0.001. But if
    # your background has moving trees and stuff, use a higher value,
    # maybe start with 0.01.
    # 2) firstFrame: This is the first frame from the video/webcam.
    def __init__(self, alpha, firstFrame):
        self.alpha = alpha
        self.backGroundModel = firstFrame

    def getForeground(self, frame):
        # apply the background averaging formula:
        # NEW_BACKGROUND = CURRENT_FRAME * ALPHA + OLD_BACKGROUND * (1 - APLHA)
        self.backGroundModel = frame * self.alpha + self.backGroundModel * (1 - self.alpha)

        # after the previous operation, the dtype of
        # self.backGroundModel will be changed to a float type
        # therefore we do not pass it to cv2.absdiff directly,
        # instead we acquire a copy of it in the uint8 dtype
        # and pass that to absdiff.

        return cv2.absdiff(self.backGroundModel.astype(np.uint8), frame)




# Just a simple function to perform
# some filtering before any further processing.
def denoise(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return frame

path_video = 'video/'
video_file = 'area1.mov'

video = cv2.VideoCapture(path_video + video_file)

ret, frame = video.read()

fborders = ip.fborders




video_h, video_w = frame.shape[:2]
video_k = 2
video_w = int(video_w/video_k)
video_h = int(video_h/video_k)
frame = cv2.resize(frame, (video_w, video_h))

field_mask = np.zeros(frame.shape[:2], dtype="uint8")
cv2.fillPoly(field_mask,[np.array(fborders)], 255, 1)

# ------- hist 1 ------------
masked_field = cv2.bitwise_and(frame, frame, mask=field_mask)
k_crop = 5

#  ------------- team 1 -----------
team1_init_box = (149, 132, 47, 48)
# team1_init_box = cv2.selectROI('team1 player', frame, False)
# print(team1_init_box)
# cv2.destroyWindow("team1 player")

team1_init_crop = frame[int(team1_init_box[1]):int(team1_init_box[1]+team1_init_box[3]),
                 int(team1_init_box[0]):int(team1_init_box[0]+team1_init_box[2])]

team1_init_crop_big = cv2.resize(team1_init_crop, (k_crop * team1_init_crop.shape[1],
                                       k_crop * team1_init_crop.shape[0]), interpolation = cv2.INTER_CUBIC)

team1_init_box = (69, 42, 29, 74)
# team1_init_box = cv2.selectROI('team1 color', team1_init_crop_big, False)
# print(team1_init_box)
# cv2.destroyWindow("team1 color")

team1_init_crop_color = team1_init_crop_big[int(team1_init_box[1]):int(team1_init_box[1]+team1_init_box[3]),
                 int(team1_init_box[0]):int(team1_init_box[0]+team1_init_box[2])]

team1_hsv = cv2.cvtColor(team1_init_crop_color, cv2.COLOR_BGR2HSV)

team1_hist = cv2.calcHist([team1_hsv],[0, 1], None, [180, 256], [0, 180, 0, 256])
# plt.plot(team1_hist)
# plt.show()

# normalize histogram and apply backprojection
cv2.normalize(team1_hist, team1_hist, 0, 255, cv2.NORM_MINMAX)

# mask_team1 = filter_by_hist(masked_field, team1_hist)
# ------- end hist 1 --------



if ret is True:
    backSubtractor = BackGroundSubtractor(0.2, denoise(frame)) # 0.2
    run = True
else:
    run = False

while True: #(run):
    # Read a frame from the camera
    ret, frame = video.read()

    # If the frame was properly read.
    if not ret:
        break

    frame1 = frame.copy()
    frame1 = cv2.resize(frame1, (video_w, video_h))

    frame1 = cv2.bitwise_and(frame1, frame1, mask=field_mask)


    # Show the filtered image
    # frame2 = denoise(frame1)

    # get the foreground
    foreGround = backSubtractor.getForeground(denoise(frame1))

    foreGround = cv2.cvtColor(foreGround.copy(), cv2.COLOR_BGR2GRAY)

    # Apply thresholding on the background and display the resulting mask
    ret, mask = cv2.threshold(foreGround, 15, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # edged = cv2.Canny(fgmask, 30, 100)  # any gradient between 30 and 150 are considered edges
    mask_dilated = cv2.dilate(mask, kernel, iterations=2) # dilate

    mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    frame3 = cv2.bitwise_and(frame1, frame1, mask=mask_closed)

    # cv2.imshow('mask_closed', mask_closed)

    # ----------try hist 1 ------------
    mask_team1 = filter_by_hist(frame3, team1_hist)
    cv2.imshow('try1', np.hstack([frame3, mask_team1]))
    # --------end try hist 1 ----------

    # ------------- find conts ----------------

    edged = cv2.Canny(frame3, 30, 100)  # any gradient between 30 and 150 are considered edges
    # edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(cnts):
        # if the contour is too small, ignore it

        M = cv2.moments(cnts[i])
        if M['m00'] == 0:
            pass
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        if cv2.contourArea(c) > 200 and cv2.contourArea(c) < 2000: # and cx < 150 and cy > 150:

            (x1, y1, w1, h1) = cv2.boundingRect(c)
            # if h1 >= w1:
            cv2.rectangle(frame1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

            cv2.putText(frame1, str(cv2.contourArea(c)), (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    cv2.imshow('frame1', frame1)
    cv2.imshow('edged', edged)
    # ---------------- end conts ----------------


    # Note: The mask is displayed as a RGB image, you can
    # display a grayscale image by converting 'foreGround' to
    # a grayscale before applying the threshold.
    # cv2.imshow('mask', np.hstack([mask, mask_closed]))
    # cv2.imshow('mask', mask)

    # cv2.imshow('frames', np.hstack([frame1, frame3]))

    # cv2.imshow(np.hstack([frame1, mask]))

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break

cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()

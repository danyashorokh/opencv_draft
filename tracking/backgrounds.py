import numpy as np
import cv2
import init_points as ip


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


fgbg1 = cv2.createBackgroundSubtractorMOG2()



if ret is True:
    backSubtractor = BackGroundSubtractor(0.2, denoise(frame))
    run = True
else:
    run = False

while True: #(run):
    # Read a frame from the camera
    ret, frame = video.read()

    if ret is True:

        frame1 = frame.copy()
        frame1 = cv2.resize(frame1, (video_w, video_h))
        frame1 = cv2.bitwise_and(frame1, frame1, mask=field_mask)

        # If the frame was properly read.

        # Show the filtered image


        frame2 = denoise(frame1)


        # ------- class 1 ------

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

        # ----- end class 1 ------

        # ----------- fgmask ----------

        frame4 = cv2.GaussianBlur(frame1, (3, 3), 0)

        fgmask = fgbg1.apply(frame4.copy())
        _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

        masks = cv2.bitwise_and(mask, fgmask, mask=mask_closed)

        # ---------- end fgmask -----------

        cv2.imshow('mask', np.hstack([masks, mask, fgmask]))
        # cv2.imshow('mask', mask)

        # cv2.imshow('frame', np.hstack([frame1, frame3]))

        # cv2.imshow(np.hstack([frame1, mask]))

        key = cv2.waitKey(10) & 0xFF
    else:
        break

    if key == 27:
        break

cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()
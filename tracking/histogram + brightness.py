import cv2
import numpy as np

def filter_by_hist(field, obj_hist):

    field_hsv = cv2.cvtColor(field.copy(), cv2.COLOR_BGR2HSV)

    obj_by_hist = cv2.calcBackProject([field_hsv], [0, 1], obj_hist, [0, 180, 0, 256], 1)

    # # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv2.filter2D(obj_by_hist, -1, disc, obj_by_hist)

    ret, thresh = cv2.threshold(obj_by_hist, 50, 255, 0)
    thresh = cv2.merge((thresh, thresh, thresh))

    team_mask = cv2.bitwise_and(field, thresh)

    return team_mask

def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

path = 'image/'

# image = cv2.imread(path + 'foot.png')
# image = cv2.imread(path + 'h2.jpeg')
image = cv2.imread(path + 'wide2.png')



image = increase_brightness(image, value=50)




image_h, image_w = image.shape[:2]
image_k = 1.4
image_w = int(image_w/image_k)
image_h = int(image_h/image_k)

image = cv2.resize(image, (image_w, image_h))


box = cv2.selectROI('roi', image, False)
print(box)
cv2.destroyWindow('roi')

roi = image[int(box[1]):int(box[1]+box[3]),
                 int(box[0]):int(box[0]+box[2])]

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

target = image.copy()
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

# calculating object histogram
roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

# normalize histogram and apply backprojection
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)

res = filter_by_hist(target, roihist)




res = np.vstack((target,res))
# cv2.imwrite('res.jpg',res)


cv2.imshow('res', res)
cv2.waitKey(0)

cv2.destroyAllWindows()
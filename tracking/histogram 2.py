import cv2
import numpy as np

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

path = 'image/'

# image = cv2.imread(path + 'foot.png')
image = cv2.imread(path + 'h2.jpeg')


image_h, image_w = image.shape[:2]
image_k = 2
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

# ----- color quantization -----
def kmeans(image, k=3):
    img = image.copy()

    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2

# res = np.vstack((target,res))
# cv2.imwrite('res.jpg',res)

res2 = kmeans(res, k=2)


cv2.imshow('res', np.hstack([res, res2]))
cv2.waitKey(0)

cv2.destroyAllWindows()
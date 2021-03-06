import pandas as pd
import numpy as np
# import rasterio
import math
import cv2

import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

import queue as queue

from shapely.geometry import Point

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# фильтр для бинарных изображений по размеру сегментов (реализован на основе волнового алгоритма)
# лучше воспользоваться функциями remove_small_objects и remove_small_holes из библиотеки skimage
def filterByLength(input, length, more=True):

    copy = input.copy()
    output = np.zeros_like(input)

    for i in range(input.shape[0]):

        for j in range(input.shape[1]):
            if (copy[i][j] == 255):

                q_coords = queue.Queue()
                output_coords = list()

                copy[i][j] = 100

                q_coords.put([i, j])
                output_coords.append([i, j])

                while (q_coords.empty() == False):

                    currentCenter = q_coords.get()

                    for idx1 in range(3):
                        for idx2 in range(3):

                            offset1 = - 1 + idx1
                            offset2 = - 1 + idx2

                            currentPoint = [currentCenter[0] + offset1, currentCenter[1] + offset2]

                            if (currentPoint[0] >= 0 and currentPoint[0] < input.shape[0]):
                                if (currentPoint[1] >= 0 and currentPoint[1] < input.shape[1]):
                                    if (copy[currentPoint[0]][currentPoint[1]] == 255):
                                        copy[currentPoint[0]][currentPoint[1]] = 100

                                        q_coords.put(currentPoint)
                                        output_coords.append(currentPoint)

                if (more == True):
                    if (len(output_coords) >= length):
                        for coord in output_coords:
                            output[coord[0]][coord[1]] = 255
                else:
                    if (len(output_coords) < length):
                        for coord in output_coords:
                            output[coord[0]][coord[1]] = 255

    return output


# сегментация тени при помощи порогового преобразования
def shadowSegmentation(roi, threshold=50):

    thresh = cv2.equalizeHist(roi)

    hist = cv2.resize(thresh, (int(thresh.shape[1] * 5), int(thresh.shape[0] * 5)))
    # cv2.waitKey(0)

    ret, thresh = cv2.threshold(thresh, threshold, 255, cv2.THRESH_BINARY_INV)

    thresh2 = cv2.resize(thresh, (int(thresh.shape[1] * 5), int(thresh.shape[0] * 5)))

    if np.count_nonzero(thresh) > thresh.shape[1] ** 2 / 8:

        tmp = filterByLength(thresh, 60)

    elif np.count_nonzero(thresh) < thresh.shape[1] ** 2 / 10:

        # kernel = np.ones((3, 3), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        tmp = cv2.dilate(thresh, kernel, iterations=1)

    else:
        tmp = thresh

    print(np.count_nonzero(tmp), thresh.shape[1] ** 2)

    if np.count_nonzero(tmp) != 0:
        thresh = tmp


    thresh3 = cv2.resize(tmp, (int(tmp.shape[1] * 5), int(tmp.shape[0] * 5)))

    shadow = np.hstack((hist, thresh2, thresh3))
    cv2.imshow('hist + threshold', shadow)
    # cv2.waitKey(0)


    return thresh


# определяем размер(длину) тени; x,y - координаты здания на изображении thresh
def getShadowSize(thresh, x, y, roi_x, roi_y, img, neighbors):

    # определяем минимальную дистанцию от здания до пикселей тени
    min_dist = thresh.shape[0]
    min_dist_coords = (0, 0) # 0 0

    y = int(thresh.shape[0]/2)
    x = int(thresh.shape[1]/2)

    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if (thresh[i, j] == 255) and (math.sqrt((i - y) * (i - y) + (j - x) * (j - x)) < min_dist):
                min_dist = math.sqrt((i - y) * (i - y) + (j - x) * (j - x))
                min_dist_coords = (i, j)  # y, x

    print('min_dist = %s' % min_dist)

    if min_dist <= thresh.shape[0]/4:

        # определяем сегмент, который содержит пиксель с минимальным расстоянием до здания
        import queue as queue

        q_coords = queue.Queue()
        q_coords.put(min_dist_coords)

        mask = thresh.copy()

        # cv2.imshow('shadow0', mask)
        # cv2.waitKey(0)

        output_coords = list()
        output_coords.append(min_dist_coords)

        while q_coords.empty() == False:
            currentCenter = q_coords.get()

            for idx1 in range(3):
                for idx2 in range(3):

                    offset1 = - 1 + idx1
                    offset2 = - 1 + idx2

                    currentPoint = [currentCenter[0] + offset1, currentCenter[1] + offset2]

                    if (currentPoint[0] >= 0 and currentPoint[0] < mask.shape[0]):
                        if (currentPoint[1] >= 0 and currentPoint[1] < mask.shape[1]):
                            if (mask[currentPoint[0]][currentPoint[1]] == 255):
                                mask[currentPoint[0]][currentPoint[1]] = 100
                                img[currentPoint[0]+roi_y][currentPoint[1]+roi_x] = [0,0,255]
                                q_coords.put(currentPoint)
                                output_coords.append(currentPoint)

                                # all changed 0 to 1

        # отрисовываем ближайшую тень
        mask = np.zeros_like(mask)

        for i in range(len(output_coords)):
            mask[output_coords[i][0]][output_coords[i][1]] = 255
            # img[currentPoint[0] + roi_y][currentPoint[1] + roi_x] = [0, 0, 255]

        # cv2.imshow('red shadow', img)
        cv2.waitKey(0)

        img1 = mask.copy()

        # cv2.imshow('shad', mask)
        # cv2.waitKey(0)

        # find contours in the edge map
        cnts = cv2.findContours(img1.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] #if imutils.is_cv2() else cnts[1]

        img2 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None

        # print(len(cnts), '-----------')
        # loop over the contours individually
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            # if cv2.contourArea(c) < 10000:
            #     continue

            # compute the rotated bounding box of the contour
            # orig = thresh.copy()
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(img2, [box.astype("int")], -1, (0, 255, 0), 1)

            # loop over the original points and draw them
            for (x, y) in box:
                # cv2.circle(img2, (int(y), int(x)), 2, (0, 0, 255), -1)

                # unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                # compute the midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # draw the midpoints on the image
                cv2.circle(img2, (int(tltrX), int(tltrY)), 1, (255, 0, 0), -1)
                cv2.circle(img2, (int(blbrX), int(blbrY)), 1, (255, 0, 0), -1)
                # cv2.circle(img2, (int(tlblX), int(tlblY)), 1, (255, 0, 0), -1)
                # cv2.circle(img2, (int(trbrX), int(trbrY)), 1, (255, 0, 0), -1)

                # draw lines between the midpoints
                cv2.line(img2, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                         (255, 0, 255), 1)
                # cv2.line(img2, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                #          (255, 0, 255), 1)

                # compute the Euclidean distance between the midpoints
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                if dA == 0: dA = 1
                # dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                # draw the object sizes on the image
                # cv2.putText(img2, "{:.1f}in".format(dA),
                #             (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.15, (255, 255, 255), 2)
                # cv2.putText(img2, "{:.1f}in".format(dB),
                #             (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.15, (255, 255, 255), 2)

            print('dA = %s' % (str(dA)))

            print('dB = %s' % (str(dB)))

            scale_img2 = img2.copy()
            scale_img2 = cv2.resize(scale_img2, (int(scale_img2.shape[1] * 5), int(scale_img2.shape[0] * 5)))
            cv2.imshow('img points', scale_img2)
            # cv2.waitKey(0)
            cv2.imwrite(dir + "img2.png", scale_img2)

        dA = int(dA)


        # определяем размер(длину) тени при помощи морфологической операции erode
        kernel = np.ones((3, 3), np.uint8)


        i = 0
        while np.count_nonzero(mask) != 0:
            mask = cv2.erode(mask, kernel, iterations=1)
            i += 1

            # cv2.imshow('shadow erode = ' + str(i), mask)
            # cv2.waitKey(0)

        print('i = %s' % (str(i)))

        # correct result for small objects
        if i == 1:

            dA = max(dA, dB)

        if dA == 0 and dB == 0:
            dA = 1


        # correct result for shadows with neighbours
        if neighbors > 0 and dA / i > 2:
            dA = i + 1

    else:
        dA = 1


    return dA


# определяем область, где нет облаков
def getNoCloudArea(b, g, r, n):
    gray = (b + g + r + n) / 4.0

    band_max = np.max(gray)

    gray = np.around(gray * 255.0 / band_max).astype(np.uint8)
    gray[gray == 0] = 255

    ret, no_cloud_area = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((50, 50), np.uint8)
    no_cloud_area = cv2.morphologyEx(no_cloud_area, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((100, 100), np.uint8)

    no_cloud_area = cv2.morphologyEx(no_cloud_area, cv2.MORPH_DILATE, kernel)
    no_cloud_area = cv2.morphologyEx(no_cloud_area, cv2.MORPH_DILATE, kernel)
    no_cloud_area = 255 - no_cloud_area

    return no_cloud_area




dir = 'task2/'
dir_pan = dir + 'tif/pan.tif'
dir_swir = dir + 'tif/swir.tif'
dir_mul = dir + 'tif/mul.tif'

# df = pd.read_csv(dir + 'csv/train.csv')
df = pd.read_csv(dir + 'csv/test.csv')

tr = False

df = df[df['img_name'] == 'swir']
df.index = range(len(df))

# читаем csv в формате lat,long,height
# df = pd.read_csv('buildings.csv')
# читаем csv файл с информацией о снимке
# image_df = pd.read_csv('geotiff.csv')

# угол солнца над горизонтом
sun_elevation = 47.7
y_pix = 3.949
x_pix = 4.168
sun_az = 159.8
sat_az = 4.77

# m_in_pix = y_pix / abs(math.cos(math.degrees(180 - (sun_az)))) #/1.2

m_in_pix = 11.57


print('m_in_pix = %s' % m_in_pix)


swir_bn = cv2.imread(dir_swir, 0)
swir_bn = cv2.GaussianBlur(swir_bn, (7, 7), 0)
swir_bn = cv2.adaptiveThreshold(swir_bn,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 2)

src = cv2.imread(dir_swir)
# src = rasterio.open(dir_swir)


# with rasterio.open(dir_swir) as src:

# геокоординаты снимка
# в этом случаем снимок сохранён в ПСК epsg:32637
# left = src.bounds.left
# right = src.bounds.right
# bottom = src.bounds.bottom
# top = src.bounds.top

height, width = src.shape[0], src.shape[1]

b, g, r = cv2.split(src)

# берём зелёный канал, т.к. он самый контрастный
band_max = g.max()
img = np.around(g * 255.0 / band_max).astype(np.uint8)

# определяем маску области, где нет облаков
no_cloud_area = getNoCloudArea(b, g, r, 3)


swirs = np.hstack((img * 2, swir_bn))
# cv2.imshow('swirs', swirs)
# cv2.waitKey(0)
# cv2.imwrite(dir + "img_g.png", swirs)

img_points = img.copy()
img_points = 2 * img_points
img_points = cv2.cvtColor(img_points, cv2.COLOR_GRAY2BGR)

heights = list()

# определяем тень в окне размером (size, size)
size = 25

df = df.loc[:35:, :]
print(df.head())
print(df.tail())

# df.index = range(len(df))
# print(df)

lst = []

for idx in range(df.shape[0]):

    # геокоординаты и высота здания
    x = df.loc[idx]['x']
    y = df.loc[idx]['y']

    print('-------- %s ----------' % idx)

    if tr:
        build_height = int(df.loc[idx]['height'])

    # переводим геокоординаты в координаты плоской системы координат epsg:32637
    # (в ПСК можно выполнять линейную интерполяцию и таким образом находить координаты зданий уже на самом изображении)
    # build_coords = Point(get32637(lon, lat))

    # координаты снимка и зданий в одной ПСК, поэтому можно определить координаты здания уже на самом изображении
    # x, y = crsToPixs(width, height, left, right, bottom, top, build_coords)

    # ищем тень зданий, если в этом месте нет облаков
    if no_cloud_area[y][x] == 255:
        # зная азимут солнца, мы знаем в каком направлении должны быть тени зданий
        # поиск тени зданий будем выполнять в этом направлении
        # roi = img[y - size:y, x - size:x].copy()

        roi = img[y - int(size/2):y+int(size/2), x - int(size/2):x + int(size/2)].copy()


        # roi = img[x - size:y, x - int(size/2):x+int(size/2)].copy()

        roi_y = y - int(size/2)
        roi_x = x - int(size/2)
        # cv2.imwrite(dir + "roi.png", roi)

        # cv2.imshow('roi', roi)
        cv2.waitKey(0)


        img0 = img.copy()
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img0 = 2 * img0
        cv2.circle(img0, (x, y), 2, (0, 0, 255), -1)

        cv2.circle(img_points, (x, y), 2, (0, 0, 255), -1)

        # cv2.rectangle(img0, (y, x), (y-size, x-size), (255, 0, 0), 1)
        cv2.rectangle(img0, (int(x-size/2), int(y-size/2)), (int(x+size/2), int(y+size/2)), (255, 0, 0), 1)

        if tr:
            cv2.putText(img0, str(build_height), (int(x-size/2), int(y-size/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(img0, str(idx), (int(x + size/2), int(y - size / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (65, 215, 245), 1, cv2.LINE_AA)


        shadow = shadowSegmentation(roi, threshold=40)
        # cv2.imshow('shadow', shadow)
        cv2.waitKey(0)

        # cv2.imshow('shadow detect', img0)
        cv2.waitKey(0)

        # проверяем наличие построек рядом, для корректного анализа тени
        neighbors = 0
        for i in range(len(df)):
            if i != idx:
                if abs(x - df.loc[i, 'x']) < size/4 and abs(y - df.loc[i, 'y']) < size/4:
                    neighbors += 1

        print('neighbors = %s' % neighbors)



        # (size, size) - координаты здания в roi
        # умножаем длину тени в пикселях на размер пикселя в метрах
        # (в этом случае пространственное разрешение 3 м)
        shadow_length = int(getShadowSize(shadow, size, size, roi_x, roi_y, img0, neighbors) * m_in_pix) # 3

        print(shadow_length)

        lst.append([idx, shadow_length])

        cv2.putText(img0, str(shadow_length), (int(x - size / 2), int(y + size / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

        cv2.namedWindow('img 1 circle') #, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('img 1 circle', img0)
        #
        # cv2.waitKey(0)

        cv2.imwrite(dir + "img_points_mul.png", img0)


        # est_height = shadow_length * math.tan(sun_elevation * 3.14 / 180)
        # est_height = int(est_height)

        # heights.append((est_height, build_height))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

cv2.imwrite(dir + "img_points.png", img_points)


# MAPE = 0
#
# for i in range(len(heights)):
#     MAPE += (abs(heights[i][0] - heights[i][1]) / float(heights[i][1]))
#
# MAPE *= (100 / float(len(heights)))
#
# print(MAPE)
#
# print(heights)

test = pd.DataFrame(lst, columns=['id', 'height'])
test.to_csv('test.csv')
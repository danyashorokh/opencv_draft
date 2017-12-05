import pandas as pd
import numpy as np
import rasterio
import math
import cv2

from shapely.geometry import Point


# фильтр для бинарных изображений по размеру сегментов (реализован на основе волнового алгоритма)
# лучше воспользоваться функциями remove_small_objects и remove_small_holes из библиотеки skimage
def filterByLength(input, length, more=True):
    import queue as queue

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


# переводим координаты из плоской системы координат epsg:32637 в координаты географической системы координат epsg:4326
def getLongLat(x1, y1):
    from pyproj import Proj, transform

    inProj = Proj(init='epsg:32637')
    outProj = Proj(init='epsg:4326')

    x2, y2 = transform(inProj, outProj, x1, y1)

    return x2, y2


# переводим координаты из географической системы координат epsg:4326 в координаты плоской системы координат epsg:32637
def get32637(x1, y1):
    from pyproj import Proj, transform

    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:32637')

    x2, y2 = transform(inProj, outProj, x1, y1)

    return x2, y2


# переводим координаты из плоской системы координат epsg:32637 в координаты изображения(пиксели)
def crsToPixs(width, height, left, right, bottom, top, coords):
    x = coords.xy[0][0]
    y = coords.xy[1][0]

    x = width * (x - left) / (right - left)
    y = height - height * (y - bottom) / (top - bottom)

    x = int(math.ceil(x))
    y = int(math.ceil(y))

    return x, y


# сегментация тени при помощи порогового преобразования
def shadowSegmentation(roi, threshold=60):
    thresh = cv2.equalizeHist(roi)


    ret, thresh = cv2.threshold(thresh, threshold, 255, cv2.THRESH_BINARY_INV)

    tmp = filterByLength(thresh, 50)

    if np.count_nonzero(tmp) != 0:
        thresh = tmp

    return thresh


# определяем размер(длину) тени; x,y - координаты здания на изображении thresh
def getShadowSize(thresh, x, y):
    # определяем минимальную дистанцию от здания до пикселей тени
    min_dist = thresh.shape[0]
    min_dist_coords = (0, 0)

    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if (thresh[i, j] == 255) and (math.sqrt((i - y) * (i - y) + (j - x) * (j - x)) < min_dist):
                min_dist = math.sqrt((i - y) * (i - y) + (j - x) * (j - x))
                min_dist_coords = (i, j)  # y,x

    # определяем сегмент, который содержит пиксель с минимальным расстоянием до здания
    import queue as queue

    q_coords = queue.Queue()
    q_coords.put(min_dist_coords)

    mask = thresh.copy()
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
                            q_coords.put(currentPoint)
                            output_coords.append(currentPoint)

    # отрисовываем ближайшую тень
    mask = np.zeros_like(mask)
    for i in range(len(output_coords)):
        mask[output_coords[i][0]][output_coords[i][1]] = 255

    # определяем размер(длину) тени при помощи морфологической операции erode
    kernel = np.ones((3, 3), np.uint8)
    i = 0
    while np.count_nonzero(mask) != 0:
        mask = cv2.erode(mask, kernel, iterations=1)
        i += 1

    return i + 1


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

df = pd.read_csv(dir + 'csv/train.csv')
df = df[df['img_name'] == 'swir']
df.index = range(len(df))

# читаем csv в формате lat,long,height
# df = pd.read_csv('buildings.csv')
# читаем csv файл с информацией о снимке
# image_df = pd.read_csv('geotiff.csv')

# угол солнца над горизонтом
sun_elevation = 47.7


swir_bn = cv2.imread(dir_swir, 0)
swir_bn = cv2.GaussianBlur(swir_bn, (7, 7), 0)
swir_bn = cv2.adaptiveThreshold(swir_bn,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 2)


src = rasterio.open(dir_swir)

# with rasterio.open(dir_swir) as src:

# геокоординаты снимка
# в этом случаем снимок сохранён в ПСК epsg:32637
left = src.bounds.left
right = src.bounds.right
bottom = src.bounds.bottom
top = src.bounds.top

height, width = src.shape

b, g, r, n = map(src.read, (1, 2, 3, 4))

# берём зелёный канал, т.к. он самый контрастный
band_max = g.max()
img = np.around(g * 255.0 / band_max).astype(np.uint8)

# определяем маску области, где нет облаков
no_cloud_area = getNoCloudArea(b, g, r, n)


swirs = np.hstack((img * 2, swir_bn))
cv2.imshow('swirs', swirs)
cv2.waitKey(0)
cv2.imwrite(dir + "img_g.png", swirs)

heights = list()

# определяем тень в окне размером (size, size)
size = 30

for idx in range(0, df.shape[0]):

    # геокоординаты и высота здания
    x = df.loc[idx]['x']
    y = df.loc[idx]['y']
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
        roi = img[y - size:y, x - size:x].copy()
        shadow = shadowSegmentation(roi)

        # (size, size) - координаты здания в roi
        # умножаем длину тени в пикселях на размер пикселя в метрах
        # (в этом случае пространственное разрешение 3 м)
        shadow_length = getShadowSize(shadow, size, size) * 4

        est_height = shadow_length * math.tan(sun_elevation * 3.14 / 180)
        est_height = int(est_height)

        heights.append((est_height, build_height))

MAPE = 0

for i in range(len(heights)):
    MAPE += (abs(heights[i][0] - heights[i][1]) / float(heights[i][1]))

MAPE *= (100 / float(len(heights)))

print(MAPE)

print(heights)
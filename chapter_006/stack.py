import cv2
import numpy as np

def imStack(imgArr, scale=1):
    imgList = []
    for i in range(0, imgArr.__len__()):
        imgList.append([])

    type = 0

    for x in imgArr:
        for y in x:
            if y.shape.__len__() > 2:
                if y.shape[2] > type:
                    type = y.shape[2]

    i = 0
    for x in imgArr:
        for y in x:
            if not y.shape.__len__() > 2:
                imgTemp = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
                imgList[i].append(cv2.resize(imgTemp, (int(imgTemp.shape[1]*scale), int(imgTemp.shape[0]*scale))))
            else:
                imgList[i].append(cv2.resize(y, (int(y.shape[1]*scale), int(y.shape[0]*scale))))
        i += 1

    imgStacked = []

    for x in imgList:
        imgStacked.append(np.hstack(x))

    if imgStacked.__len__() == 1:
        return imgStacked[0]
    else:
        return np.vstack(imgStacked)
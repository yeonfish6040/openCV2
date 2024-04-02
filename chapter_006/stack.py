import cv2
import numpy as np

def imStack(imgArr, scale=1):
    imgList = []
    for i in range(0, imgArr.__len__()):
        imgList.append([])

    type = 0

    shape = (int(imgArr[0][0].shape[1]*scale), int(imgArr[0][0].shape[0]*scale))

    imgCountMax = 0

    for x in imgArr:
        for y in x:
            if y.shape.__len__() > 2:
                if y.shape[2] > type:
                    type = y.shape[2]
            else:
                if not type > 2:
                    type = 2
        if imgCountMax < x.__len__():
            imgCountMax = x.__len__()

    i = 0
    for x in imgArr:
        j = 0
        for y in x:
            if not (lambda y: (lambda y: True if y.shape[2] != 2 else False) if y.shape.__len__() > 2 else False)(y) and type > 2:
                imgTemp = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
                imgList[i].append(cv2.resize(imgTemp, shape))
            else:
                imgList[i].append(cv2.resize(y, shape))
            j += 1
        if j < imgCountMax:
            for k in range(0, imgCountMax-j):
                imgTemp = np.zeros((lambda type, shape: (shape[1], shape[0], type) if type > 2 else (shape[1], shape[0]))(type, shape), np.uint8)
                cv2.line(imgTemp, (0, 0), (imgTemp.shape[1], imgTemp.shape[0]), (255, 255, 0), 2) # X
                cv2.line(imgTemp, (imgTemp.shape[1], 0), (0, imgTemp.shape[0]), (255, 255, 0), 2) # X

                # ㅁ
                cv2.line(imgTemp, (0, 0), (0, imgTemp.shape[0]), (255, 255, 255), 1)
                cv2.line(imgTemp, (0, imgTemp.shape[0]), (imgTemp.shape[1], imgTemp.shape[0]), (255, 255, 255), 1)
                cv2.line(imgTemp, (imgTemp.shape[1], imgTemp.shape[0]), (imgTemp.shape[1], 0), (255, 255, 255), 1)
                cv2.line(imgTemp, (imgTemp.shape[1], 0), (0, 0), (255, 255, 255), 1)
                imgTemp = (lambda type, img: img if type > 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))(type, imgTemp)
                imgList[i].append(cv2.resize(imgTemp, (int(imgTemp.shape[1]), int(imgTemp.shape[0]))))
        i += 1

    imgStacked = []

    for x in imgList:
        imgStacked.append(np.hstack(x))

    if imgStacked.__len__() == 1:
        return imgStacked[0]
    else:
        return np.vstack(imgStacked)
import cv2
import numpy as np

def crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y + h, x:x + w]
    return rect

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
                cv2.line(imgTemp, (0, 0), (imgTemp.shape[1], imgTemp.shape[0]), (255, 255, 0), 2)
                cv2.line(imgTemp, (imgTemp.shape[1], 0), (0, imgTemp.shape[0]), (255, 255, 0), 2)
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

class contourUtil:

    def __init__(self, img):
        self.img = img
        self.imgContour = img.copy()
        self.color = (0, 255, 0)
        self.lineWidth = 1
        self.minSize = 5
        self.contoursList = []

    def setMinSize(self, minSize):
        self.minSize = minSize
    def getContours(self, img, shape=False):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.minSize:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                objCor = len(approx)
                x, y, w, h = cv2.boundingRect(approx)
                self.contoursList.append([approx, [x, y, w, h]])

        return self.contoursList

    def drawContours(self, img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.minSize:
                cv2.drawContours(self.imgContour, cnt, -1, self.color, self.lineWidth)

        return self.imgContour

    def drawRect(self, img):
        imgRect = self.drawContours(img)
        i = 0
        for approx, (x, y, w, h) in self.getContours(img):
            imgRect = cv2.rectangle(imgRect, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imgRect, str(i), (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            i += 1
        return imgRect
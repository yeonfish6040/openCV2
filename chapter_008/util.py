import cv2
import numpy

class contourUtil:

    def __init__(self, img):
        self.img = img
        self.imgContour = img.copy()
        self.color = (0, 255, 0)
        self.lineWidth = 2
    def drawContours(self, img, shape=False):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                cv2.drawContours(self.imgContour, cnt, -1, self.color, self.lineWidth)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                objCor = len(approx)
                print(objCor)
                x, y, w, h = cv2.boundingRect(approx)

                type = ""

                if shape:
                    if objCor == 3: type = "Tri"
                    if objCor == 4: type = "Rect"
                    if type == "Rect" and ((w-h)**2)**(1/2) < 4: type = "Square"
                    if type == "": type = "Circles"

                cv2.rectangle(self.imgContour, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(self.imgContour, type, (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    def getContours(self):
        return self.imgContour
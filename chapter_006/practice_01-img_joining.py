import cv2
import numpy as np
from stack import imStack

# chapter 002
webcam = cv2.VideoCapture(0)
kernel = np.ones((5, 5), np.uint8)

while webcam.isOpened():
    ret, flame = webcam.read()

    zoomRange = 200

    img = flame
    img2 = flame[int(img.shape[0]/2)-zoomRange:int(img.shape[0]/2)+zoomRange, int(img.shape[0]/2)-zoomRange:int(img.shape[1]/2)+zoomRange]

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (15, 15), 0)
    imgCanny = cv2.Canny(img, 80, 80)
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
    imgEroded = cv2.erode(imgDilation, kernel, iterations=1)

    # imgHor = np.hstack((imgGray, imgBlur, imgCanny, imgDilation, imgEroded))
    imgHor = imStack([[img, imgGray, imgBlur, imgCanny, imgDilation, imgEroded]])

    imgGray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    imgBlur2 = cv2.GaussianBlur(imgGray2, (15, 15), 0)
    imgCanny2 = cv2.Canny(img2, 80, 80)
    imgDilation2 = cv2.dilate(imgCanny2, kernel, iterations=1)
    imgEroded2 = cv2.erode(imgDilation2, kernel, iterations=1)

    imgHor2 = np.hstack((imgGray2, imgBlur2, imgCanny2, imgDilation2, imgEroded2))

    cv2.imshow("Output - stack", imgHor)
    cv2.imshow("Output - bigStack", imgHor2)
    if cv2.waitKey(1) == ord("q"):
        break
import cv2
import numpy as np

webcam = cv2.VideoCapture(0)
kernel = np.ones((5, 5), np.uint8)

while webcam.isOpened():
    ret, flame = webcam.read()

    img = flame

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (15, 15), 0)
    imgCanny = cv2.Canny(img, 100, 100)
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
    imgEroded = cv2.erode(imgDilation, kernel, iterations=1)

    cv2.imshow("Output - gray", imgGray)
    cv2.imshow("Output - grayBlur", imgBlur)
    cv2.imshow("Output - canny", imgCanny)
    cv2.imshow("Output - cannyDilation", imgDilation)
    cv2.imshow("Output - cannyEroded", imgEroded)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
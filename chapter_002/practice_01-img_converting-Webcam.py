import cv2
import numpy as np
from chapter_006.stack import imStack

webcam = cv2.VideoCapture(0)
kernel = np.ones((5, 5), np.uint8)

while webcam.isOpened():
    ret, flame = webcam.read()

    img = flame

    show = []

    show.append([])
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show[0].append(imgGray)
    imgBlur = cv2.GaussianBlur(imgGray, (15, 15), 0)
    show[0].append(imgBlur)
    imgCanny = cv2.Canny(img, 100, 100)
    show[0].append(imgCanny)
    show.append([])
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
    show[1].append(imgDilation)
    imgEroded = cv2.erode(imgDilation, kernel, iterations=1)
    show[1].append(imgEroded)

    cv2.imshow("Output - all", imStack(show))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
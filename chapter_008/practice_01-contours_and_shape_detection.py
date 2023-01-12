import cv2
from util import contourUtil
import numpy as np

from chapter_006.stack import imStack

path = "../resources/img/shapes.png"
# path = "../resources/img/cards.png"
img = cv2.imread(path)
shows = [[]]
shows[0].append(img)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 200, 200)
utilContour = contourUtil(img)
utilContour.drawContours(imgCanny, True)
shows[0].append(imgGray)
shows[0].append(imgBlur)
shows[0].append(imgCanny)
shows.append([])
shows[1].append(utilContour.getContours())


resultImg = imStack(shows, 1)
cv2.imshow("Output - all", resultImg)
cv2.waitKey(0)
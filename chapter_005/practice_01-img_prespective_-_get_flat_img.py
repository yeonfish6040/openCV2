import numpy as np

import cv2
import numpy as py

img = cv2.imread("../resources/img/cards.png")

width, height = 250, 350
points = [732, 142], [767, 223], [648, 208], [617, 123];
pts1 = np.float32([points])
# pts1 = np.float32([[0, 0], [0, 10], [10, 10], [10, 0]])
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOut = cv2.warpPerspective(img, matrix, (width, height))

cv2.line(img, points[0], points[1], (0, 255, 0), 2)
cv2.line(img, points[1], points[2], (0, 255, 0), 2)
cv2.line(img, points[2], points[3], (0, 255, 0), 2)
cv2.line(img, points[3], points[0], (0, 255, 0), 2)

cv2.imshow("Output - raw", img)
cv2.imshow("Output - perspective", imgOut)
cv2.waitKey(0)
import sys
import time
import cv2
import numpy as np
import util

img = cv2.imread("./resource/diff.jpeg")
img1 = img[:, :int(img.shape[1]/2)]
img2 = img[:, int(img.shape[1]/2)-7:]

a = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
b = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
a = cv2.Canny(a, 500, 500)
b = cv2.Canny(b, 500, 500)

# a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)

rows, cols = a.shape

def loop():
    global img1, img2, a, b

    imgGray = np.zeros((a.shape[0], a.shape[1], 3), np.uint8)

    for x in range(rows):
        for y in range(cols):
            if not a[x, y] == b[x, y]:
                imgGray[x, y] = (255, 255, 255)
                # print(x, y)

    imgGray = cv2.cvtColor(imgGray, cv2.COLOR_BGR2GRAY)

    au = util.contourUtil(img1)
    bu = util.contourUtil(img2)
    au.setMinSize(cv2.getTrackbarPos("minSize", "TrackBar"))
    bu.setMinSize(cv2.getTrackbarPos("minSize", "TrackBar"))

    cv2.imshow("Different Point", util.imStack([[au.drawRect(imgGray), bu.drawRect(imgGray), imgGray]]))
    if cv2.waitKey(500) == ord('q'):
        sys.exit()
    else:
        loop()

def TrackbarHandling(a):
    pass

cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar", 1080, 240)
cv2.createTrackbar("minSize", "TrackBar", 5, 500, TrackbarHandling)

loop()
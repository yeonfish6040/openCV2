import cv2
import numpy as np
from chapter_006.stack import imStack

values = {"hmin": 0, "hmax": 46, "smin": 53, "smax": 255, "vmin": 165, "vmax": 255}

def onTrackBarChange(x):
    global values
    values = {
        "hmin": cv2.getTrackbarPos("HueMin", "TrackBar"),
        "hmax": cv2.getTrackbarPos("HueMax", "TrackBar"),
        "smin": cv2.getTrackbarPos("SatMin", "TrackBar"),
        "smax": cv2.getTrackbarPos("SatMax", "TrackBar"),
        "vmin": cv2.getTrackbarPos("ValMin", "TrackBar"),
        "vmax": cv2.getTrackbarPos("ValMax", "TrackBar")
    }

    print("\n"*20)
    for i in values.keys():
        print(i+": "+str(values[i]))

    path = "../resources/img/toyCar.png"
    img = cv2.imread(path)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lowerLimit = np.array([values["hmin"], values["smin"], values["vmin"]])
    upperLimit = np.array([values["hmax"], values["smax"], values["vmax"]])
    mask = cv2.inRange(imgHSV, lowerLimit, upperLimit)

    imgResult = cv2.bitwise_and(img, img, mask=mask)

    all = imStack([[img, imgHSV], [mask, imgResult]], 1)

    cv2.imshow("Output - all", all)



cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar", 640, 240)
cv2.createTrackbar("HueMin", "TrackBar", 0, 179, onTrackBarChange)
cv2.createTrackbar("HueMax", "TrackBar", 46, 179, onTrackBarChange)
cv2.createTrackbar("SatMin", "TrackBar", 53, 255, onTrackBarChange)
cv2.createTrackbar("SatMax", "TrackBar", 255, 255, onTrackBarChange)
cv2.createTrackbar("ValMin", "TrackBar", 165, 255, onTrackBarChange)
cv2.createTrackbar("ValMax", "TrackBar", 255, 255, onTrackBarChange)

onTrackBarChange(1)

cv2.waitKey(0)
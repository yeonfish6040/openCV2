import cv2
import json
import numpy as np
from chapter_006.stack import imStack

f = open('../project_004/color.json', 'r')
data = json.load(f)
f.close()
values = {"hmin": data[0][0], "hmax": data[1][0], "smin": data[0][1], "smax": data[1][1], "vmin": data[0][2], "vmax": data[1][2]}

cam = cv2.VideoCapture(0)
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
    sc, img = cam.read()

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    with open("../project_004/color.json", "w") as f:
        f.write("[[%s, %s, %s], [%s, %s, %s]]" % (values["hmin"], values["smin"], values["vmin"], values["hmax"], values["smax"], values["vmax"]))
        f.close()

    lowerLimit = np.array([values["hmin"], values["smin"], values["vmin"]])
    upperLimit = np.array([values["hmax"], values["smax"], values["vmax"]])
    mask = cv2.inRange(imgHSV, lowerLimit, upperLimit)

    imgResult = cv2.bitwise_and(img, img, mask=mask)

    all = imStack([[img, imgHSV], [mask, imgResult]], 0.3)

    cv2.imshow("Output - all", all)



cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar", 640, 240)
cv2.createTrackbar("HueMin", "TrackBar", data[0][0], 179, onTrackBarChange)
cv2.createTrackbar("HueMax", "TrackBar", data[1][0], 179, onTrackBarChange)
cv2.createTrackbar("SatMin", "TrackBar", data[0][1], 255, onTrackBarChange)
cv2.createTrackbar("SatMax", "TrackBar", data[1][1], 255, onTrackBarChange)
cv2.createTrackbar("ValMin", "TrackBar", data[0][2], 255, onTrackBarChange)
cv2.createTrackbar("ValMax", "TrackBar", data[1][2], 255, onTrackBarChange)

while True:
    # print("\n" * 20)
    # for i in values.keys():
    #     print(i + ": " + str(values[i]))

    sc, img = cam.read()

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lowerLimit = np.array([values["hmin"], values["smin"], values["vmin"]])
    upperLimit = np.array([values["hmax"], values["smax"], values["vmax"]])
    mask = cv2.inRange(imgHSV, lowerLimit, upperLimit)

    imgResult = cv2.bitwise_and(img, img, mask=mask)

    all = imStack([[img, imgHSV], [mask, imgResult]], 0.3)

    cv2.imshow("Output - all", all)
    cv2.waitKey(1)

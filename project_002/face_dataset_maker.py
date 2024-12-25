# conda create -n tensorflow python=3.9
# conda activate tensorflow
# python -m pip install matplotlib
# python -m pip install tensorflow-macos==2.10
# python -m pip install tensorflow-metal
# python -m pip install tensorflow-macos==2.12
# python -m pip install keras_applications
# python -m pip install opencv-python

import cv2
import os
from chapter_006.stack import imStack

upperbodyCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_upperbody.xml")
faceCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_eye.xml")
cam = cv2.VideoCapture(0)

# type = "validation"
type = "faces"

# uid = input("Enter your ID: ")
uid = "0"
if not os.path.exists(type+"/" + str(uid)):
    os.makedirs(type+"/" + str(uid))

trainList = os.listdir(type+"/"+uid)

i = 0
for e in trainList:
    if e.find(".jpg") != -1:
        if int(e.split(".")[0]) > i:
            i = int(e.split(".")[0])
imgList = []
def loop():
    global i
    global imgList

    sc, img = cam.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.2, 4)

    faceList = []

    for (x, y, w, h) in faces:
        eyes = eyeCascade.detectMultiScale(imgGray[y:y+h, x:x+w], 1.2, 6)
        if eyes.__len__() != 2:
            continue

        cv2.putText(img, "front_face", (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(img, "front_face", (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        faceList.append(img[y:y+h, x:x+w])

        imgList.append(img[y:y+h, x:x+w])
        try:
            cv2.imwrite(type+"/" + str(uid) + "/" + str(i) + ".jpg", img[y:y + h, x:x + w], [cv2.IMWRITE_JPEG_QUALITY, 100])
        except Exception as e:
            pass
        print("\rNum: " + str(i), end="")
        i += 1


    cv2.imshow("Result", img)
    if faceList.__len__() > 0:
        cv2.imshow("faces", imStack([faceList]))
    cv2.waitKey(1)
    loop()

loop()
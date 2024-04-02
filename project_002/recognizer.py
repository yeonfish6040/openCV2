import cv2
import os

import numpy as np

from chapter_006.stack import imStack

trainList = os.listdir("trains")
trainList.sort(reverse=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trains/%s' % trainList[0])

faceCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_frontalface_default.xml")

# names = ["YeonJun", "HyunJun", "Rina", "MinHye", "JaeWon"]
# names = ["YeonJun", "Hyeyeon"]
names = ["YeonJun", "Yeonsu"]


cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.4, 1)

    imgList = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(imgGray[y:y + h, x:x + w])

        if confidence < 55:
            id = names[id-1]
            imgList.append(img[y:y + h, x:x + w])

        else:
            id = "unknown"

        confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
    cv2.imshow("Result", img)
    cv2.waitKey(1)

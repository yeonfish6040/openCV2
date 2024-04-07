import cv2
from chapter_006.stack import imStack

eyeCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_eye.xml")
cam = cv2.VideoCapture(0)

def loop():

    sc, img = cam.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = eyeCascade.detectMultiScale(imgGray, 1.2, 12)

    eyeList = []

    for (x, y, w, h) in faces:
        cv2.putText(img, "eye", (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(img, "eye", (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        eyeList.append(img[y:y+h, x:x+w])


    cv2.imshow("Result", img)
    if eyeList.__len__() > 0:
        cv2.imshow("eyes", imStack([eyeList], 4))
    cv2.waitKey(1)
    loop()

loop()
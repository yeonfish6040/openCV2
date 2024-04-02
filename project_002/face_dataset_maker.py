import cv2
import os
from chapter_006.stack import imStack
from skimage.metrics import structural_similarity

upperbodyCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_upperbody.xml")
faceCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_eye.xml")
cam = cv2.VideoCapture(0)

uid = input("Enter your ID: ")


trainList = os.listdir("faces")
trainList.sort(reverse=True)
filteredTrainList = []
i = 0
for s in trainList:
    if "face_%s" % uid in s:
        i += 1
i += 1
imgList = []
def loop():
    global i
    global imgList

    sc, img = cam.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.2, 4)

    faceList = []

    for (x, y, w, h) in faces:
        cv2.putText(img, "front_face", (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(img, "front_face", (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        faceList.append(img[y:y+h, x:x+w])

        imgList.append(img[y:y+h, x:x+w])
        try:
            cv2.imwrite("faces/face_" + str(uid) + "_" + str(i) + ".png", img[y:y + h, x:x + w])
        except Exception as e:
            pass
        i += 1


    cv2.imshow("Result", img)
    if faceList.__len__() > 0:
        cv2.imshow("faces", imStack([faceList]))
    cv2.waitKey(1)
    loop()

loop()
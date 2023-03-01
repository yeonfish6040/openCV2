import cv2
from chapter_006.stack import imStack
from skimage.metrics import structural_similarity

upperbodyCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_upperbody.xml")
faceCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_eye.xml")
cam = cv2.VideoCapture(0)
i = 0
imgList = []
def loop():
    global i
    global imgList

    sc, img = cam.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    upperBody = upperbodyCascade.detectMultiScale(imgGray, 1.1, 4)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    eyes = eyeCascade.detectMultiScale(imgGray, 1.1, 4)

    faceList = []

    # for (x, y, w, h) in upperBody:
    #     cv2.putText(img, "upper_body", (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    #     cv2.putText(img, "upper_body", (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in faces:
        cv2.putText(img, "front_face", (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(img, "front_face", (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        faceList.append(img[y:y+h, x:x+w])
        ok = True
        for face in imgList:
            faceGray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            imgGray = cv2.resize(cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY), faceGray.shape)
            if structural_similarity(faceGray, imgGray, full=True)[0]*100>60:
                ok = False
        if ok:
            imgList.append(img[y:y+h, x:x+w])
            try:
                cv2.imwrite("faces/face_"+str(i)+".png", img[y:y+h, x:x+w])
            except Exception as e:
                pass
            i += 1
    for (x, y, w, h) in eyes:
        for face in faces:
            if face[0] < x and face[1] < y and face[0]+face[2] > x+w and face[1]+face[3] > y+h:
                cv2.putText(img, "eye", (x + w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(img, "eye", (x + w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


    cv2.imshow("Result", img)
    if faceList.__len__() > 0:
        cv2.imshow("faces", imStack(faceList))
    cv2.waitKey(1)
    loop()

loop()
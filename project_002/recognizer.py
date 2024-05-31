import math
import os
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

faceCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_eye.xml")

trainList = os.listdir("trains")
i = 0
for e in trainList:
    if e.split(".")[0] != "":
        if int(e.split(".")[0].split("_")[1]) > i:
            i = int(e.split(".")[0].split("_")[1])

print('trains/train_%d.keras' % i)
model = tf.keras.models.load_model('trains/train_%d.keras' % i)


names = ['HyunJun', 'JaeWon', 'MinHye', 'Rina', 'Soonmo', 'Yeonjun']
names = ['Hayyeon', 'HyunJun', 'JaeWon', 'MinHye', 'Rina', 'Soonmo', 'Yeonjun']
# names = ["YeonJun", "Hyeyeon"]
# names = ["YeonJun", "Yeonsu"]

height, width = 260, 260
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.2, 2)

    imgList = []
    for (x, y, w, h) in faces:

        cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        isFace = False
        eyes = eyeCascade.detectMultiScale(imgGray[y:y + h, x:x + w], 1.2, 6)
        eyeList = []
        for (ex, ey, ew, eh) in eyes:
            if ey > h / 2:
                continue
            isFace = True

            if ex < w / 2:
                eye = "right eye"
            else:
                eye = "left eye"
            # cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            cv2.putText(img, eye, (x + ex + ew, y + ey), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2,
                        lineType=cv2.LINE_AA)
            cv2.putText(img, eye, (x + ex + ew, y + ey), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1,
                        lineType=cv2.LINE_AA)
            eyeList.append(img[y + ey:y + ey + eh, x + ex:x + ex + ew])
        if not isFace:
            continue
        cv2.imshow("asdf", imgGray[y:y + h, x:x + w])
        prediction = model.predict(np.array([cv2.resize(imgGray[y:y + h, x:x + w], (height, width))]))
        score = tf.nn.softmax(prediction[0])
        score_fixed = round(100 * np.max(score), 3)

        if score_fixed > 20 and np.argmax(score) < names.__len__():
            id = names[np.argmax(score)]
            imgList.append(img[y:y + h, x:x + w])
        else:
            id = "unknown"

        confidence = "  {0}%".format(score_fixed)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence, (x + 5, y + h - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)

    cv2.imshow("Result", img)
    cv2.waitKey(1)

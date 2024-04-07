import math
import os
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

faceCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_frontalface_default.xml")

trainList = os.listdir("trains")
i = 0
for e in trainList:
    if e.split(".") != "":
        if int(e.split(".")[0].split("_")[1]) > i:
            i = int(e.split(".")[0].split("_")[1])

model = tf.keras.models.load_model('trains/train_%d.keras' % i)


names = ['HyunJun', 'JaeWon', 'MinHye', 'Rina', 'Yeonjun']
# names = ["YeonJun", "Hyeyeon"]
# names = ["YeonJun", "Yeonsu"]

height, width = 540, 540
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.2, 4)

    imgList = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        prediction = model.predict(np.array([cv2.resize(cvt_image[y:y + h, x:x + w], (height, width))]))
        score = tf.nn.softmax(prediction[0])

        print(score)

        if (100 * np.max(score)) > 80:
            id = names[np.argmax(score)]
            imgList.append(img[y:y + h, x:x + w])

        else:
            id = "unknown"

        confidence = "  {0}%".format(math.floor(100 * np.max(score)))

        cv2.putText(img, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence, (x + 5, y + h - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)

    cv2.imshow("Result", img)
    cv2.waitKey(1)

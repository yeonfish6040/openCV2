import time

import cv2
import numpy as np
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from chapter_006.stack import imStack

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../resources/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
)

faceCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_eye.xml")

cam = cv2.VideoCapture(0)

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp.solutions.drawing_styles
    #     .get_default_face_mesh_contours_style())
    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
    #       landmark_drawing_spec=None,
    #       connection_drawing_spec=mp.solutions.drawing_styles
    #       .get_default_face_mesh_iris_connections_style())

    return annotated_image

with FaceLandmarker.create_from_options(options) as landmarker:
    def loop():

        sc, img = cam.read()

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(imgGray, 1.1, 10)

        eyeList = []

        for (x, y, w, h) in faces:

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            face_landmarker_result = landmarker.detect(mp_image)
            if face_landmarker_result.face_landmarks.__len__() != 0:
                print(face_landmarker_result)
                img = draw_landmarks_on_image(img, face_landmarker_result)

            eyes = eyeCascade.detectMultiScale(imgGray[y:y+h, x:x+w], 1.2, 6)
            # if eyes.__len__() != 2:
            #     continue
            for (ex, ey, ew, eh) in eyes:
                if ey > h/2:
                    continue

                if ex < w/2:
                    eye = "left eye"
                else:
                    eye = "right eye"
                # cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                cv2.putText(img, eye, (x+ex+ew, y+ey), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(img, eye, (x+ex+ew, y+ey), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                eyeList.append(img[y+ey:y+ey+eh, x+ex:x+ex+ew])


            cv2.putText(img, "face", (x + w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(img, "face", (x + w, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


        cv2.imshow("Result", img)
        # if eyeList.__len__() > 0:
        #     cv2.imshow("eyes", imStack([eyeList], 4))
        cv2.waitKey(1)
        loop()

    loop()
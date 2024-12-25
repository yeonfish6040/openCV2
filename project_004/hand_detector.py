import numpy as np
import cv2
import math
import time
import asyncio
# from websocket import create_connection


from hand_detector_utils import *

SOCKET_URL = "ws://lyj.kr:17003"
# USER = "869906936219447348"
USER = "766514069036859392"

lastSent = LastSent()
def send(lastSent, msg):
    # if lastSent.update():
    #     for i in range(1):
    #         ws = create_connection(SOCKET_URL)  # open socket
    #         ws.send(USER+"|"+msg)  # send to socket
    #         ws.close()  # close socket
    pass

last = []

good_condition = False
drawing_box = True
full_frame = False
stabilize_highest_point = True

old_highest_point = (-1, -1)

x1_crop = 0
y1_crop = 60
x2_crop = 420
y2_crop = 520

# Open Camera
try:
    default = 0  # Try Changing it to 1 if webcam not found
    capture = cv2.VideoCapture(default)
except:
    print("No Camera Source Found!")

while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()

    width = frame.shape[1]

    img_right = frame[y1_crop:y2_crop, 0:int(width / 2)]

    try:

        contour_right = detectHand(img_right)

        defects_right, drawing_right = findDefects(img_right, contour_right)

        count_defects = countDefects(defects_right, contour_right, img_right)

        highest_point = trackHighestPoint(defects_right, contour_right)

        if (stabilize_highest_point):
            if (old_highest_point == (-1, -1)):
                old_highest_point = highest_point
            else:
                diag_difference = np.linalg.norm(np.asarray(old_highest_point) - np.asarray(highest_point))

                if (diag_difference >= 9.5):
                    old_highest_point = highest_point
                else:
                    highest_point = old_highest_point;

        if (full_frame):
            highest_point = (highest_point[0], highest_point[1])
            cv2.circle(frame, highest_point, 10, [255, 0, 255], -1)
        else:
            cv2.circle(img_right, highest_point, 10, [255, 0, 255], -1)
            highest_point = (highest_point[0] + x1_crop, highest_point[1] + y1_crop)

        x, y, w, h = cv2.boundingRect(contour_right);
        if count_defects == 0 and h > w+50:
            if abs(x + w * (4 / 7) - highest_point[0]) < 50:
                cv2.putText(frame, "Fuck you", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 0, 255])
            else:
                cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 0, 255])
        else:
            textDefects(frame, count_defects, debug_var=False)

        if count_defects == 0 and abs(x + w * (4 / 7) - highest_point[0]) < 50:
            # send(lastSent, "🖕|가운데 손가락")
            pass
        elif count_defects == 1:
            send(lastSent, "✌|개 쩌는 브이")
            cv2.imwrite("V.jpeg", frame)
        elif count_defects == 4:
            send(lastSent, "✋|하이 파이브")

        if (drawing_box):
            cv2.rectangle(frame, (0, y1_crop), (int(width / 2), y2_crop), (0, 0, 255), 1)
        cv2.imshow("Full Frame", frame)

        all_image_right = np.hstack((drawing_right, img_right))
        cv2.imshow('Recognition Right', all_image_right)

        last.append(count_defects)
        if (len(last) > 5):
            last = last[-5:]
            # last = []

        if (good_condition):
            pass

    except Exception as e:
        # print(e)
        pass

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
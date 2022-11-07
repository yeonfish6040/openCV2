import cv2

webcam = cv2.VideoCapture(0)

while cv2.waitKey(1) & 0xFF != ord("q"):
    if webcam.isOpened() != True:
        continue
    success, flame = webcam.read()
    if not success:
        continue
    cv2.imshow("Webcam", flame)
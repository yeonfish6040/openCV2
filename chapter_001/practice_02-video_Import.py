import cv2

cap = cv2.VideoCapture("../resources/videos/test.mov")

i = 0

while True:
    i = 0
    success, frame = cap.read()
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
import cv2

img = cv2.imread("../resources/img/img.png")

cv2.imshow("Output", img)
cv2.waitKey(0)
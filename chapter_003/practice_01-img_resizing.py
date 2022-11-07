import cv2
import numpy as np

img = cv2.imread("../resources/img/img.png")
kernel = np.ones((5, 5), np.uint8)

print(img.shape)

imgResized = cv2.resize(img, (300, 300))
imgCropped = img[0:100, 0:300]

cv2.imshow("Output - raw", img)
cv2.imshow("Output - resized", imgResized)
cv2.imshow("Output - cropped", imgCropped)
cv2.waitKey(0)
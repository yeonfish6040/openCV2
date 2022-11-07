import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)
# img[100:300, 100: 300] = 255, 255, 0

# diagonal
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (255, 255, 0), 3)
cv2.line(img, (img.shape[1], 0), (0, img.shape[0]), (255, 255, 0), 3)

# middle rectangle
cv2.rectangle(img, (int(img.shape[1]/2)-100, int(img.shape[0]/2)-100), (int(img.shape[1]/2)+100, int(img.shape[0]/2)+100), (255, 255, 0), 3)

# middle circle
cv2.circle(img, (int(img.shape[1]/2), int(img.shape[0]/2)), int((lambda x: x[0] if (x[0] < x[1]) else x[1])(img.shape)/2), (255, 255, 0), 3)

# text
cv2.putText(img, "Open CV2", (int(img.shape[1]/2)-80, int(img.shape[0]/2)), cv2.FONT_ITALIC, 1, (255, 255, 0), 2)

# show
cv2.imshow("Output - drawing ", img)
cv2.waitKey(0)
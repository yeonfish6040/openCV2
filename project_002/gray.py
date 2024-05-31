import cv2
import os

for i in os.listdir("./validation"):
    if "DS" in i:
        continue
    for j in os.listdir("./validation/"+i):
        if "DS" in j:
            continue
        cv2.imwrite("./validation/"+i+"/"+j, cv2.cvtColor(cv2.imread("./validation/"+i+"/"+j), cv2.COLOR_BGR2GRAY))
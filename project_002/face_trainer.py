import time
# import cv2
import numpy as np
from PIL import Image
import os
import torch
import torch.nn.functional as Functional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




faces, ids = getImagesAndLabels('faces')

startTime = time.time()

print("Training... (it can take a few minutes)")

torch.tensor()

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.train(faces,np.array(ids))

trainList = os.listdir("trains")
trainList.sort(reverse=True)
if trainList.__len__() == 0:
    trainList.append("_-1")
recognizer.write('trains/train_%d.yml' % (int((trainList[0]).split("_")[1].split(".")[0]) + 1))

print("Finished! Time: ", time.time()-startTime)
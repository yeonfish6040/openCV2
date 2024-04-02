import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torch.nn.functional as F

class FaceDataSet(torch.utils.data.Dataset):
      def __init__(self):
          self.path = "faces"
          self.x_data, self.y_data = self.__getImagesAndLabels__()

      def __len__(self):
          return self.x_data.__len__()

      def __getitem__(self, idx):
          x = torch.FloatTensor(self.x_data[idx])
          y = torch.FloatTensor(self.y_data[idx])
          return x, y

      def __getImagesAndLabels__(self):
          imagePaths = os.listdir(self.path)

          faceSamples = []
          ids = []
          print("Loading images...", end="")
          i = 0
          for imagePath in imagePaths:
              if imagePath.find(".png") == -1:
                  i += 1
                  continue
              PIL_img = Image.open(os.path.join(self.path, imagePath)).convert('L')
              faceSamples.append(np.array(PIL_img, 'uint8'))
              ids.append(int(imagePath.split("_")[1]))
              i += 1
              print("\rLoading images... %d/%d" % (i, imagePaths.__len__()), end="")
          print()
          return faceSamples, ids


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 10 * 10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 10 * 10)
        output = self.fc1(output)

        return output
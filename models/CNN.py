from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet, self).__init__()
        # Architecture Le-Net 5
        self.conv1 = nn.Conv2d(3, 6, 5)  # (input_channels, output_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)  # (kernel_size, stride)
        self.conv2 = nn.Conv2d(6, 16,
                               5)  # input of this layer must be the same as the output of the previous layer conv layer
        self.flatten = nn.Flatten()
        # 120 and 84 is the size of feature map
        # 16 * 9 * 9 is the dimension of input after being flatten
        self.fc1 = nn.Linear(16 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Input shape ([64, 3, 48, 48])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = x.view(-1, 16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

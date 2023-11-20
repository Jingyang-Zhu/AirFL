import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from os.path import dirname, abspath, join
from torch.autograd import Variable
from tqdm import tqdm
from option import args_parser
# class CNN(nn.Module):
#
#     def __init__(self, input_channels, output_channels):
#         super(CNN, self).__init__()
#         self.fc1 = nn.Linear(28*28, output_channels)
#
#     def forward(self, x):
#         x = x.contiguous().view(-1, 28*28) # [, 28*28]
#         x = self.fc1(x) # [, 10]
#         return x

class CNN(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_channels)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # [, 10, 12, 12]
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2) # [, 20, 4, 4]
        x = x.contiguous().view(-1, 320) # [, 320]
        x = F.relu(self.fc1(x)) # [, 50]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x) # [, 10]
        return x
#
class CNNCifar(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_channels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = CNN(input_channels = 1,output_channels = 10)
    test_data = torch.rand(5, 1, 28, 28)
    test_outputs = model(test_data)
    for name, param in model.named_parameters():
        print(name)
    print("Output size:", test_outputs.size())
    print('parameters:', sum(param.numel() for param in model.parameters()))
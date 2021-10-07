import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

import common

model_urls = {
        "resnet50" : "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        }

class ResNet50(nn.Module):
    def __init__(self, pretrained, num_input_channel, num_output):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True, num_classes=1000)
        self.resnet50.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_output)
    
    def forward(self, x):
        x = self.resnet50(x.float())
        return x

class ResNet50LSTM(nn.Module):
    def __init__(self, pretrained, num_input_channel, num_output):
        super(ResNet50LSTM, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True, num_classes=1000)
        self.resnet50.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 1000)
        self.fc1 = nn.Linear(1000, 300)

        # TODO
        # nn.LSTM(input_size, hidden_size, sequence_length)
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, batch_first=True, num_layers=3, bidirectional=False, dropout=0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_output)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.resnet50(x.float())
        x = self.fc1(x)

        out, (h_n, c_n) = self.lstm(x.unsqueeze(1)) # x = (batch_size, 1, 300)
        x = self.fc2(out[-1, :, :])
        x = F.relu(x)
        x = self.fc3(x)
        return x

class ResNet18LSTM(nn.Module):
    def __init__(self, pretrained, num_input_channel, num_output):
        super(ResNet18LSTM, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True, num_classes=1000)
        self.resnet18.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Sequential(nn.Linear(self.resnet18.fc.in_features, 1000))
        self.fc1 = nn.Linear(1000, 300)

        # nn.LSTM(input_size, hidden_size, sequence_length)
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 100)

    def forward(self, x):
        hidden = None
        batch_size = x.shape[0]
        for t in range(x.size(1)):
            with torch.no_grad():
                res = self.resnet18(x[:, t, :, :, :].float())
                res = self.fc1(res)
            out, (h_n, c_n) = self.lstm(res.unsqueeze(0), hidden)
        x = self.fc2(out[-1, :, :])
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.reshape(x, (batch_size, -1, 2))
        return x

class LSTM(nn.Module):
    def __init__(self, pretrained, num_input_channel, num_output, batch_size):
        super(LSTM, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 10, kernel_size=3, stride=1, padding = 1, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(10*22*40, 1000)
        self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(common.IMG_H*common.IMG_W*num_input_channel, 1000)
        # batch_first=True -> (batch_size, sequence_length, input_feature_vector_size)
        # batch_first=False -> (sequence_length, batch_size, input_feature_vector_size)
        self.lstm = nn.LSTM(1000, 100, batch_first=True, num_layers=4, bidirectional=True, dropout=0.3)
        self.fc2 = nn.Linear(100, num_output)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x.float())
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc(x)
#         x = self.flatten(x)
#         x = self.fc1(x.float())
        x = x.view(batch_size, 1, 1000)
        o, (h_n, c_n) = self.lstm(x)
        x = self.fc2(h_n[-1, :, :])
        return x

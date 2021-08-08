import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url

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
        self.fc1 = nn.Linear(1000, 100)

        # TODO
        # nn.LSTM(input_size, hidden_size, sequence_length)
        self.lstm = nn.LSTM(100, 10, batch_first=True, num_layers=4, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(10, num_output)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.resnet50(x.float())
        x = self.fc1(x)

        x = x.view(batch_size, 1, 100)
        x = x.permute(1, 0, 2) # (len_sequence, batch_size, vector_size)
        out, (h_n, c_n) = self.lstm(x)
#        h_n = self.dropout(h_n)
        x = self.linear(h_n[-1, :, :])
        return x, out

import torch
import torch.nn as nn

SEQ_LEN = 100
NUM_FEATURES = 12

class CNN(nn.Module):
    def __init__(self, input_channels=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2,1))
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2,1))
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2,2))
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2,2))
        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.randn(1, input_channels, SEQ_LEN, NUM_FEATURES)
            x = self.pool1(self.bn1(self.conv1(dummy)))
            x = self.pool2(self.bn2(self.conv2(x)))
            x = self.pool3(self.bn3(self.conv3(x)))
            x = self.pool4(self.bn4(self.conv4(x)))
            flat_size = x.flatten(1).shape[1]
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

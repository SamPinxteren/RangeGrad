import torch.nn as nn
import minmax.mm as mm

class Convolutional(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = mm.Conv2d(3, 6, 5)
        self.pool = mm.MaxPool2d(2, 2)
        self.conv2 = mm.Conv2d(6, 16, 5)
        self.fc1 = mm.Linear(16 * 5 * 5, 120)
        self.fc2 = mm.Linear(120, 84)
        self.fc3 = mm.Linear(84, 10)
        self.softmax = mm.LogSoftMax
        self.relu = mm.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.relu(x))
        
        x = self.conv2(x)
        x = self.pool(self.relu(x))
        
        x = mm.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        
        return self.softmax(x)

import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        self.conv1 = nn.Conv1d(1, 6, kernel_size=3)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=3)
        self.conv2_drop = nn.Dropout1d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, X):
        X = self.conv1(X)


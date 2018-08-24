import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, vocab_size, word_dim):
        self.conv1_3 = nn.Conv2d(300, 1, kernel_size=(300, 3), stride=(1, 0))
        self.conv1_5 = nn.Conv2d(300, 1, kernel_size=(300, 5), stride=(1, 0))
        self.conv1_7 = nn.Conv2d(300, 1, kernel_size=(300, 7), stride=(1, 0))

        nn.MaxPool2d()

        self.conv2_drop = nn.Dropout2d(1
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.embeddings = nn.Embedding(vocab_size, word_dim)

    def forward(self, X):

        n_data, n_word = X.size()

        X = self.embeddings(X)

        X = X.view(n_data, 1, 1, n_word)

        X_3 = nn.MaxPool2d(torch.relu(self.conv1_3(X)))
        

        X_5 = self.conv1_5(X)
        X_7 = self.conv1_7(X)
        


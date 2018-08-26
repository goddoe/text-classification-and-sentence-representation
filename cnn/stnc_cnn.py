import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceCNN(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(SentenceCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.word_win_size = [3, 5, 7]

        self.conv_list = nn.ModuleList(
                            [nn.Conv2d(1, 1, kernel_size=(w, embed_dim))
                             for w in self.word_win_size])

        self.embeddings = nn.Embedding(vocab_size,
                                       embed_dim)

        self.output_dim = len(self.word_win_size)

    def forward(self, X):
        n_data, n_word = X.size()

        X = self.embeddings(X)

        X = X.view(n_data, 1, n_word, self.embed_dim)

        C = [F.relu(conv(X)) for conv in self.conv_list]

        C_hat = torch.stack([F.max_pool2d(
                                c, c.size()[2:]).squeeze()
                             for c in C], dim=1)

        return C_hat

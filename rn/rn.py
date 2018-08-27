import torch
import torch.nn as nn
import torch.nn.functional as F


class RN(nn.Module):
    """Implementation of Relation Network
    """

    def __init__(self, vocab_size, embed_dim):
        super(RN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        self.linear_l = nn.Linear(embed_dim, embed_dim)
        self.linear_r = nn.Linear(embed_dim, embed_dim)
        self.relation = nn.Linear(embed_dim, embed_dim)

        self.output_dim = embed_dim

    def forward(self, X):
        """Embed sequence and get word distribution of prediction.

        Args:
            X: inputs, shape of (batch_size, sequence)

        Returns:
            tensor, Sentence representation
        """

        batch_size, n_sentence = X.size()

        X_embed = self.embeddings(X)  # (batch_size x sentence x embed_dim)

        X_i = X_embed.unsqueeze(1).repeat(1, n_sentence, 1, 1)
        X_j = X_embed.unsqueeze(2).repeat(1, 1, n_sentence, 1)

        X_combi = torch.cat([X_i, X_j], dim=3)
        


from itertools import product

import torch.nn as nn
import torch.nn.functional as F


class RN(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(RN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        self.linear_l = nn.Linear(embed_dim, embed_dim)
        self.linear_r = nn.Linear(embed_dim, embed_dim)
        self.relation = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        """Embed sequence and get word distribution of prediction.

        Args:
            X: inputs, shape of (batch_size, sequence)

        Returns:
            tensor, Sentence representation
        """
        X_embed = self.embeddings(X)

        X_embed_tuple_list = product(X_embed)

        N = 0
        stnc_repr = 0

        for X_embed_l, X_embed_r in X_embed_tuple_list:
            rn = self.relation(
                    F.relu(
                        self.linear_l(X_embed_l) + self.linear_r(X_embed_r)))
            N += 1
            stnc_repr = stnc_repr + (rn-stnc_repr)/N  # Incremental mean

        return stnc_repr

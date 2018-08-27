from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


class RN(nn.Module):

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
        X_embed = self.embeddings(X)

        X_embed_tuple_list = []
        for elem in X_embed:
            X_embed_tuple_list.append(product(elem, elem))

        stnc_repr_list = []
        for row in X_embed_tuple_list:
            N = 0
            stnc_repr = 0
            for X_embed_l, X_embed_r in row:
                rn = self.relation(
                        F.relu(
                            self.linear_l(X_embed_l) + self.linear_r(X_embed_r)))
                N += 1
                stnc_repr = stnc_repr + (rn-stnc_repr)/N  # Incremental mean
            stnc_repr_list.append(stnc_repr)

        return torch.stack(stnc_repr_list, dim=0)

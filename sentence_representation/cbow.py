import torch
import torch.nn as nn


class CBOW(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size (int): size of vocabulary.
            embed_dim (int): dimension of embedding.
        """
        super(CBOW, self).__init__()

        self.output_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, X):
        """Feed-forward CBOW.

        Args:
            X (torch.Tensor): inputs, shape of (batch_size, sequence).

        Returns:
            torch.Tensor, Sentence representation.
        """
        # (batch_size, sequence) -> (batch_size, sequence, embedding)
        X_embeded = self.embeddings(X)
        stnc_repr = torch.mean(X_embeded, dim=1)

        return stnc_repr

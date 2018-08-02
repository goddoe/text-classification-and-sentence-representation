import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size (int): size of vocabulary.
            embed_dim (int): dimension of embedding.
        """
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, X):
        """Embed sequence and get word distribution of prediction.

        Args:
            X: inputs, shape of (batch_size, sequence)

        Returns:
            tensor, word distribution
        """

        # (batch_size, sequence) -> (batch_size, sequence, embedding)
        X_embeded = self.embeddings(X)
        X_sum_sequence = torch.mean(X_embeded, dim=1)
        word_dist = F.log_softmax(self.linear_proj(X_sum_sequence), dim=1)

        return word_dist



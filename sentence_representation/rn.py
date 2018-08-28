import torch
import torch.nn as nn


class RN(nn.Module):
    """Sentence Representation with Relation Network.
    """

    def __init__(self, vocab_size, embed_dim, max_len=100):
        """
        Args:
            vocab_size (int): size of vocabulary.
            embed_dim (int): dimension of embedding.
            max_len (int): max length of sequence.
        """
        super(RN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.output_dim = embed_dim

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        self.linear_l = nn.Linear(embed_dim, embed_dim)
        self.linear_r = nn.Linear(embed_dim, embed_dim)
        self.linear_rn = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        """Feed-forward Relation Network.

        Args:
            X (torch.Tensor): inputs, shape of (batch_size, sequence).

        Returns:
            torch.Tensor, Sentence representation.
        """

        X = X[:, :self.max_len]

        batch_size, n_sentence = X.size()

        # batch_size x n_sentence x embed_dim
        X_embed = self.embeddings(X)

        # (batch_size * n_sentence) x n_sentence x embed_dim
        X_flat = X_embed.view(batch_size*n_sentence, self.embed_dim)

        # batch_size x 1 x n_sentence x embed_dim
        X_l = self.linear_l(X_flat).view(batch_size, 1, n_sentence, -1)

        # batch_size x n_sentence x 1 x embed_dim
        X_r = self.linear_r(X_flat).view(batch_size, n_sentence, 1, -1)

        # batch_size x n_sentence x n_sentence x embed_dim
        X_l = X_l.expand(-1, n_sentence, -1, -1)
        X_r = X_r.expand(-1, -1, n_sentence, -1)

        stnc_repr = self.linear_rn(torch.relu(X_l + X_r)).sum((1, 2))

        return stnc_repr

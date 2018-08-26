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

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output_dim = embed_dim

    def forward(self, X):
        """Embed sequence and get word distribution of prediction.

        Args:
            X: inputs, shape of (batch_size, sequence)

        Returns:
            tensor, Sentence representation
        """

        # (batch_size, sequence) -> (batch_size, sequence, embedding)
        X_embeded = self.embeddings(X)
        stnc_repr = torch.mean(X_embeded, dim=1)

        return stnc_repr



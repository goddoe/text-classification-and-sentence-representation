import torch.nn as nn


class Classifier(nn.Module):
    """A classifier, arbitary graph, on the top of sentence representation.

    Attributes:
        sr_model: A sentence representation module.
        input_dim: Input dimension of the classifier. Input_dim is set with sr_model output.
        output_dim: Output dimension of the model.
    """
    def __init__(self, sr_model, output_dim, vocab_size, embed_dim, **kwargs):
        """Initialization of the classifier.
        
        Args:
            sr_model (torch.nn.Module): A sentence representation module.
            output_dim (int): Output dimension of the model.
            vocab_size (int): The size of vocabulary.
            embed_dim (int): The word embedding dimension.
        """
        super(Classifier, self).__init__()

        self.sr_model = sr_model(vocab_size=vocab_size,
                                 embed_dim=embed_dim,
                                 **kwargs)

        self.input_dim = self.sr_model.output_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, X):
        return self.fc(self.sr_model(X))



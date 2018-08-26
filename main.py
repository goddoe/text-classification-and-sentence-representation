import torch
import torch.nn as nn
from cbow.cbow import CBOW
from rn.rn import RN
from cnn.stnc_cnn import SentenceCNN
from datasets import get_IMDB_iter
from train_helper import train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ======================================
# Get data
batch_size = 16
d = get_IMDB_iter(batch_size=batch_size,
                  device=device,
                  flag_use_pretrained=True)


# ======================================
# A classifier, arbitary graph, on the top of sentence representation.
class Classifier(nn.Module):
    def __init__(self, sr_model, output_dim, vocab_size, embed_dim):
        super(Classifier, self).__init__()

        self.sr_model = sr_model(vocab_size=vocab_size,
                                 embed_dim=embed_dim)

        self.input_dim = self.sr_model.output_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, X):
        return self.fc(self.sr_model(X))


# ======================================
# Sentence Representation using CBOW
clf_cbow = Classifier(sr_model=CBOW,
                      output_dim=2,
                      vocab_size=d.vocab_size,
                      embed_dim=d.embed_dim)
clf_cbow.to(device)
clf_cbow.sr_model.embeddings.weight.data.copy_(d.embeddings)

train(model=clf_cbow,
      train_iter=d.train_iter,
      test_iter=d.test_iter,
      n_epoch=5,
      lr=0.001)


# ======================================
# Sentence Representation using RN
clf_rn = Classifier(sr_model=RN,
                    output_dim=2,
                    vocab_size=d.vocab_size,
                    embed_dim=d.embed_dim)
clf_rn.to(device)
clf_rn.sr_model.embeddings.weight.data.copy_(d.embeddings)

train(model=clf_rn,
      train_iter=d.train_iter,
      test_iter=d.test_iter,
      n_epoch=5,
      lr=0.001)


# ======================================
# Sentence Representation using CNN
# Implementation of the sentence representation method in the paper
# 'Convolutional Neural Networks for Sentence Classification', Yoonkim et al.
clf_stnc_cnn = Classifier(sr_model=SentenceCNN,
                          output_dim=2,
                          vocab_size=d.vocab_size,
                          embed_dim=d.embed_dim)
clf_stnc_cnn.to(device)
clf_stnc_cnn.sr_model.embeddings.weight.data.copy_(d.embeddings)

train(model=clf_stnc_cnn,
      train_iter=d.train_iter,
      test_iter=d.test_iter,
      n_epoch=5,
      lr=0.001)



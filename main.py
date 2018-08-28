import torch
from classifier.classifier import Classifier
from sentence_representation.cbow import CBOW
from sentence_representation.rn import RN
from sentence_representation.stnc_cnn import SentenceCNN
from datasets import get_IMDB
from train_helper import train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ======================================
# Get data
batch_size = 64
d = get_IMDB(batch_size=batch_size,
             device=device,
             flag_use_pretrained=True)


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
                    embed_dim=d.embed_dim,
                    max_len=50)
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
                          embed_dim=d.embed_dim,
                          window_win_size=[3, 5, 7])
clf_stnc_cnn.to(device)
clf_stnc_cnn.sr_model.embeddings.weight.data.copy_(d.embeddings)

train(model=clf_stnc_cnn,
      train_iter=d.train_iter,
      test_iter=d.test_iter,
      n_epoch=5,
      lr=0.001)



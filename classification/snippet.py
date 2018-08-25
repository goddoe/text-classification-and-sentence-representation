import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import datasets, data
from torchtext.vocab import GloVe
from cnn.stnc_cnn import StncCNN

# Prepare datasets
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)


# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
LABEL.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits(
            (train, test), batch_size=3, device="cuda:0")

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.label)


vocab_size, word_dim = TEXT.vocab.vectors.size()

X = batch.text[0]

stnc_cnn = StncCNN(vocab_size=vocab_size,
                   word_dim=word_dim)

stnc_cnn.embeddings.weight.data.copy_(TEXT.vocab.vectors)

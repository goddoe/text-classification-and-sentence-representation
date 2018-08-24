import torch
import torch.nn as nn
from torchtext import datasets, data
from torchtext.vocab import GloVe


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


# Model
conv1_3 = nn.Conv2d(1, 1, kernel_size=(300, 3), stride=1)

conv1_5 = nn.Conv2d(1, 1, kernel_size=(300, 5), stride=1)
conv1_7 = nn.Conv2d(1, 1, kernel_size=(300, 7), stride=1)


max_pool1_3 = nn.MaxPool2d(kernel_size=(528, 298)).to("cuda:0")

vocab_size, word_dim = TEXT.vocab.vectors.size()
embeddings = nn.Embedding(vocab_size, word_dim)
embeddings.weight.data.copy_(TEXT.vocab.vectors)


X = batch.text[0]
n_data, n_word = X.size()

X_embed = embeddings(X)

X_view = X_embed.view(n_data, 1, n_word, 300)

X_conv1_3 = conv1_3(X_view)

X_max_pool1_3 = max_pool1_3(X_conv1_3)

X_final = torch.squeeze(X_max_pool1_3)


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchtext import datasets, data
from torchtext.vocab import GloVe
from cnn.stnc_cnn import SentenceCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

batch_size = 128
# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.label)


vocab_size, word_dim = TEXT.vocab.vectors.size()
batch_size = train_iter.batch_size


class ClfSentenceCNN(nn.Module):
    def __init__(self, output_dim, vocab_size, word_dim):
        super(ClfSentenceCNN, self).__init__()

        self.stnc_cnn = SentenceCNN(vocab_size=vocab_size,
                                    word_dim=word_dim)

        self.input_dim = len(self.stnc_cnn.word_win_size)
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, X):
        return self.fc(self.stnc_cnn(X))


clf_stnc_cnn = ClfSentenceCNN(output_dim=2,
                              vocab_size=vocab_size,
                              word_dim=word_dim)

clf_stnc_cnn.stnc_cnn.embeddings.weight.data.copy_(TEXT.vocab.vectors)

clf_stnc_cnn.to(device)

Y_onehot = torch.Tensor(batch_size, 2).to(device)


# train

learning_rate = 0.001
loss_list = []

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf_stnc_cnn.parameters(),
                       lr=learning_rate)

n_epoch = 5
for epoch_i in range(n_epoch):
    for batch_i, batch in enumerate(iter(train_iter)):
        X = batch.text[0]
        X_length = batch.text[1]
        Y = batch.label - 1

        clf_stnc_cnn.zero_grad()

        Y_pred = clf_stnc_cnn(X)

        loss = loss_fn(Y_pred, Y)

        loss_list.append(float(loss.to("cpu").data.numpy()))
        print(loss_list[-1])
        loss.backward()

        optimizer.step()







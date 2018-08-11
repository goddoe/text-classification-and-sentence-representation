import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import WordContextDataset

# tiny_corpus from pytorch tutorial (https://pytorch.org/tutorials/)
tiny_corpus = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold."""

# Load data
# wcd = WordContextDataset(corpus=tiny_corpus,
#                          context_size=2,
#                          min_word=1)

wcd = WordContextDataset(corpus_path="./data/alice.txt",
                         context_size=2,
                         min_word=1)


data_loader = DataLoader(wcd, batch_size=128, shuffle=True)


# Model
cbow = CBOW(vocab_size=wcd.vocab_size,
            embed_dim=100)

# Training Parameters
n_epoch = 1000
learning_rate = 0.001

optimizer = optim.SGD(cbow.parameters(),
                      lr=learning_rate)
loss_fn = nn.NLLLoss()
loss_list = []

# Use GPU, if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cbow.to(device)

for epoch_i in range(n_epoch):
    for batch_i, (X, Y) in enumerate(data_loader):
        X, Y = X.to(device), Y.to(device)
        cbow.zero_grad()

        pred_log_prob = cbow(X)

        loss = loss_fn(pred_log_prob, Y)

        loss.backward()
        loss_list.append(float(loss.to('cpu').data.numpy()))

        optimizer.step()
    print("loss : {}".format(loss))



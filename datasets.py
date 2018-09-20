from torchtext import datasets, data
from torchtext.vocab import GloVe


class DataContainer:
    def __init__(self, train_iter, test_iter, embeddings):
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.embeddings = embeddings
        self.vocab_size, self.embed_dim = self.embeddings.size()


def get_IMDB(batch_size=32,
             root=".data",
             device="cuda:0",
             flag_use_pretrained=True):
    # Prepare datasets
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False, pad_token=None, unk_token=None)

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL, root=root)

    # print information about the data
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print('vars(train[0])', vars(train[0]))

    # build the vocabulary
    if flag_use_pretrained:
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    else:
        TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits(
                (train, test), batch_size=batch_size,
                repeat=False, shuffle=True, device=device)

    vocab_size, embed_dim = TEXT.vocab.vectors.size()

    result = DataContainer(train_iter=train_iter,
                           test_iter=test_iter,
                           embeddings=TEXT.vocab.vectors)

    return result

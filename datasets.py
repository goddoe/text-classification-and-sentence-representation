from torchtext import datasets, data
from torchtext.vocab import GloVe


def get_IMDB_iter(batch_size=32,
                  root=".data",
                  device="cuda:0",
                  flag_use_pretrained=True):
    # Prepare datasets
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

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
                (train, test), batch_size=batch_size, device=device)

    vocab_size, word_dim = TEXT.vocab.vectors.size()
    
    class DataContainer:
        def __init__(self):
            self.vocab_size = vocab_size
            self. word_dim = word_dim


    return train_iter, test_iter, vocab_size, word_dim, batch_size

import torch
from torch.utils.data import Dataset


def _get_word_cnt_dict(corpus):
    """Counts words in corpus.

    Args:
        corpus (str): full text.

    Returns:
        dict, dict of word count.
    """
    word_cnt_dict = {}
    for word in corpus:
        if word in word_cnt_dict:
            word_cnt_dict[word] += 1
        else:
            word_cnt_dict[word] = 1
    return word_cnt_dict


def _to_tensor(sample):
    return (torch.tensor(sample[0], dtype=torch.long),
            torch.tensor(sample[1], dtype=torch.long))


class WordContextDataset(Dataset):
    """

    Attributes:
        corpus (list): list of corpus word.
        vocab_size (int): size of vocabulary.
        idx_word_dict (dict): dictionary which consists of
                              index as key and word as value.
        word_idx_dict (dict): dictionary which consists of
                              word as key and index as value.
    """

    def __init__(self,
                 corpus=None,
                 corpus_path=None,
                 preprocess=None,
                 transform=None,
                 context_size=2,
                 min_word=2):
        """
        Args:
            corpus_path (str): corpus txt path
            preprocess (object): preprocess to raw txt file.
            transform (object): transform function. transform must contains
                                a function of converting input to tensor.
        """
        if transform is None:
            transform = _to_tensor

        self.transform = transform
        self.preprocess = preprocess

        if corpus is None and corpus_path is None:
            raise Exception("either should be specified")

        self.corpus = corpus
        if corpus_path:
            if corpus:
                print("Ignore corpus param")
            with open(corpus_path, "rt") as f:
                self.corpus = f.read()

        if self.preprocess:
            self.corpus = self.preprocess(self.corpus)
        else:
            self.corpus = self.corpus.split()

        self.word_cnt_dict = _get_word_cnt_dict(self.corpus)
        word_cnt_tuple_list = sorted(self.word_cnt_dict.items(),
                                     key=lambda t: t[1],
                                     reverse=True)

        # Make idx_word_dict, word_idx_dict
        self.idx_word_dict = {}
        for i, (word, cnt) in enumerate(word_cnt_tuple_list):
            if cnt < min_word:
                break
            self.idx_word_dict[i] = word

        self.word_idx_dict = {word: idx
                              for idx, word in self.idx_word_dict.items()}
        self.vocab_size = len(self.idx_word_dict)

        # Make dataset
        self.data = []
        for i in range(context_size, len(self.corpus)-context_size):
            context = [self.word_idx_dict[self.corpus[i + d]]
                       for d in range(-context_size, context_size+1)
                       if d != 0]
            self.data.append((context, self.word_idx_dict[self.corpus[i]]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample



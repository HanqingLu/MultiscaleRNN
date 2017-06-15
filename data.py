import os
import cPickle as pkl
import time
import torch


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, data_path, dict, batch_size=80, maxlen=100, dict_size=-1):
        self.data = open(data_path, 'r')
        with open(dict, 'rb') as f:
            self.dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.dict_size = dict_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.data.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        data = []

        try:

            # actual work here
            while True:
                # read from data file and map to word index
                ss = self.data.readline()

                if ss == "":
                    raise IOError
                ss = unicode(ss, 'utf8')

                ss = list(ss.replace(' ', '').strip('\r\n'))

                ss = [self.dict[w] if w in self.dict else 1 for w in ss]
                if self.dict_size > 0:
                    ss = [w if w < self.dict_size else 1 for w in ss]

                # read from data file and map to word index

                if len(ss) > self.maxlen:
                    continue

                data.append(ss)

                if len(data) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(data) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return data


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        tokens = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
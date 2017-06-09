import os
import cPickle as pkl
import time


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
from model import HM_Net
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import time
import cPickle as pkl
from data import TextIterator
import torch.optim as optim
import numpy
import torch.nn.functional as Func
import os


def prepare_data(seqs_data):
    lengths_data = [len(s) for s in seqs_data]
    n_samples = len(seqs_data)
    maxlen_data = numpy.max(lengths_data) + 1

    data = numpy.zeros((maxlen_data, n_samples)).astype('int64')
    data_mask = numpy.zeros((maxlen_data, n_samples)).astype('float32')

    for idx, sentence in enumerate(seqs_data):
        data[:lengths_data[idx], idx] = sentence
        data_mask[:lengths_data[idx] + 1, idx] = 1.

    data = torch.from_numpy(data).t()
    data_mask = torch.from_numpy(data_mask).t()

    return data.cuda(), data_mask.cuda()


def save_model(path, HM_model):
    with open(path, 'wb') as f:
        torch.save(HM_model, f)


def validation(valid, HM_model):
    total_PPL = 0.0
    it = 0
    for batch in valid:
        it += 1
        inputs, mask = prepare_data(batch)
        probs = HM_model(inputs, mask)
        # print type(probs)
        PPL = 0.0
        PPL += torch.exp(probs.data/mask.sum(1)).sum(0)

        total_PPL += PPL/inputs.size(0)

    return total_PPL/it


def train(data_path=["../data/train.tok", "../data/valid.tok"], dict_path="dict.pkl",
          size_list=[512, 512], dict_size=5000, embed_size=128,
          batch_size=80, maxlen=100, learning_rate=0.1, max_epoch=100,
          valid_iter=1, show_iter=1, init='model.init.pt', reload_=True, saveto='model.pt'):

    # dict = pkl.load(open(dict_path, 'r'))
    train = TextIterator(data_path=data_path[0], dict=dict_path, batch_size=batch_size, maxlen=maxlen, dict_size=dict_size)
    valid = TextIterator(data_path=data_path[1], dict=dict_path, batch_size=batch_size, maxlen=maxlen, dict_size=dict_size)

    a = 1.0

    if reload_ and os.path.exists(init):
        print "Reloading model parameters from ", init
        with open(init, 'rb') as f:
            torch.load(f)
        print "Done"
    else:
        print "Random Initialization Because:", "reload_ = ", reload_, "path exists = ", os.path.exists(init)

    print "Build model..."
    HM_model = HM_Net(a, size_list, dict_size, embed_size)
    HM_model = HM_model.cuda()
    optimizer = optim.SGD(HM_model.parameters(), lr=learning_rate)
    print "Done"

    it = 0
    start_time = time.time()

    for epoch in range(max_epoch):
        print "start training epoch ", str(epoch)

        for batch in train:
            it += 1
            if batch is None:
                print 'Minibatch with zero sample under length ', str(maxlen)
                it -= 1
                continue
            optimizer.zero_grad()  # 0.0001s used
            inputs, mask = prepare_data(batch)  # 0.002s used
            batch_loss = HM_model(inputs, mask)  # 3-5s used
            loss = batch_loss.sum(0)
            loss.backward()  # 3s used
            optimizer.step()  # 0.001s used

            if it % show_iter == 0:
                print 'iter: ', it, ' elapse:', time.time() - start_time  # , ' train loss:', loss.data

            if it % valid_iter == 0:
                PPL = validation(valid, HM_model)
                print 'iter: ', it, ' perplexity: ', PPL
                save_path = saveto + '_iter' + str(it)
                save_model(save_path, HM_model)
                print 'iter: ', it, ' save to ', save_path


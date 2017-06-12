from model import HM_Net
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import time
import cPickle as pkl
from data import TextIterator, PTBIterator
import torch.optim as optim
import numpy
import torch.nn.functional as Func
import os
from utils import reverse


def prepare_data(seqs_data):
    lengths_data = [len(s) for s in seqs_data]
    n_samples = len(seqs_data)
    maxlen_data = numpy.max(lengths_data)  # + 1

    data = numpy.zeros((maxlen_data, n_samples)).astype('int64')
    target = numpy.zeros((maxlen_data, n_samples)).astype('int64')
    mask = numpy.zeros((maxlen_data, n_samples)).astype('float32')

    for idx, sentence in enumerate(seqs_data):
        data[:lengths_data[idx], idx] = sentence
        mask[:lengths_data[idx], idx] = 1.
        target[:lengths_data[idx], idx] = sentence[1:]+[0]
    data = torch.from_numpy(data).t()
    target = torch.from_numpy(target).t()
    mask = torch.from_numpy(mask).t()

    return data.cuda(), target.cuda(), mask.cuda()


def save_model(path, HM_model):
    with open(path, 'wb') as f:
        torch.save(HM_model, f)


def validation(valid, HM_model):
    total_PPL = 0.0
    it = 0
    for batch in valid:
        it += 1
        inputs, target, mask = prepare_data(batch)
        # print reverse(inputs[0, :], '../data/PTB/dict.pkl')
        probs = HM_model(inputs, target, mask)
        PPL = torch.exp(probs.data/mask.sum(1)).mean(0)
        total_PPL += PPL

    return (total_PPL/it).cpu().numpy()[0]


def train(data_path=["../data/train.tok", "../data/valid.tok"], dict_path="../data/dict.pkl",
          size_list=[512, 512], dict_size=5000, embed_size=128, batch_size=80, maxlen=100,
          learning_rate=0.1, clip=1, max_epoch=100,
          valid_iter=1, show_iter=1, init='model.init.pt', reload_=True, saveto='model.pt'):
    model_params = locals().copy()
    print model_params
    # dict = pkl.load(open(dict_path, 'r'))
    # train = TextIterator(data_path=data_path[0], dict=dict_path, batch_size=batch_size, maxlen=maxlen, dict_size=dict_size)
    # valid = TextIterator(data_path=data_path[1], dict=dict_path, batch_size=batch_size, maxlen=maxlen, dict_size=dict_size)
    train = PTBIterator(data_path='../data/PTB/train.txt', dict='../data/PTB/dict.pkl', batch_size=batch_size, maxlen=35)
    valid = PTBIterator(data_path='../data/PTB/valid.txt', dict='../data/PTB/dict.pkl', batch_size=batch_size, maxlen=35)

    if reload_ and os.path.exists(init):
        print "Reloading model parameters from ", init
        with open(init, 'rb') as f:
            HM_model = torch.load(f)
        print "Done"
    else:
        print "Random Initialization Because:", "reload_ = ", reload_, "path exists = ", os.path.exists(init)
        print "Build model..."
        HM_model = HM_Net(1.0, size_list, dict_size, embed_size)
        HM_model = HM_model.cuda()
        print "Done"

    PPL = validation(valid, HM_model)
    print 'start from valid perplexity: ', PPL

    optimizer = optim.SGD(HM_model.parameters(), lr=learning_rate, momentum=0.9)
    it = 0
    start_time = time.time()
    lastPPL = 100000.0
    break_flag = False

    for epoch in range(max_epoch):
        print "start training epoch ", str(epoch)
        # slope annealing trick
        HM_model.HM_LSTM.cell_1.a += 0.04
        HM_model.HM_LSTM.cell_1.a += 0.04
        for batch in train:
            it += 1
            if batch is None:
                print 'Minibatch with zero sample under length ', str(maxlen)
                it -= 1
                continue
            optimizer.zero_grad()  # 0.0001s used
            inputs, target, mask = prepare_data(batch)  # 0.002s used
            batch_loss = HM_model(inputs, target, mask)  # 3s used
            loss = batch_loss.sum(0)
            loss.backward()  # 3s used
            nn.utils.clip_grad_norm(HM_model.parameters(), clip)
            optimizer.step()  # 0.001s used

            if it % show_iter == 0:
                print 'iter: ', it, ' elapse:', time.time() - start_time, ' train loss:', loss.data

            if it % valid_iter == 0:
                PPL = validation(valid, HM_model)
                print 'iter: ', it, ' valid perplexity: ', PPL
                save_path = saveto + '_iter' + str(it)
                save_model(save_path, HM_model)
                print 'iter: ', it, ' save to ', save_path
                if abs(PPL-lastPPL) < 0.1:
                    print 'update too slow, shut down!'
                    break_flag =True
                    break

        if break_flag:
            break


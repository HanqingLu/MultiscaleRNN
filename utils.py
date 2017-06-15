from torch.autograd import Variable, Function
import torch
import torch.nn as nn
from torch import functional as F
from torch.nn import Module
import time
import cPickle as pkl
from collections import OrderedDict


def hard_sigm(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output


# def bound(x):
#     output = x > 0.5
#     return output.float()


class bound(Function):
    def forward(self, x):
        # forward : x -> output
        self.save_for_backward(x)
        output = x > 0.5
        return output.float()

    def backward(self, output_grad):
        # backward: output_grad -> x_grad
        x = self.saved_tensors
        x_grad = None

        if self.needs_input_grad[0]:
            x_grad = output_grad.clone()

        return x_grad


class masked_NLLLoss(Module):
    def __init__(self):
        super(masked_NLLLoss, self).__init__()

    def forward(self, cost, inputs, mask):
        # inputs.size = mask.size = batch_size
        # cost.size = batch_size * dict_size

        # The following version is too slow, deprecated now
        # loss = Variable(torch.zeros(inputs.size(0))).cuda()
        # for i in range(inputs.size(0)):
        #     loss[i] = - cost[i, inputs[i]] * mask[i]

        # The following version is much faster
        batch_size, dict_size = cost.size()
        cost_flat = cost.view(batch_size*dict_size)
        inputs_flat = torch.arange(0, inputs.size(0)).cuda().long()
        inputs_flat_idx = torch.mul(inputs_flat, dict_size) + inputs
        loss = -cost_flat[inputs_flat_idx] * mask
        return loss


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    data = data.cuda()
    return data


def get_batch(source, i, maxlen):
    seq_len = min(maxlen, len(source) - 1 - i)
    data = source[i:i+seq_len].t()
    target = source[i+1:i+1+seq_len].t().contiguous().view(-1)
    return data, target  # batch_size*time_steps


def reverse(sent, dict_path):
    dictionary = pkl.load(open(dict_path, 'r'))
    dict_r = OrderedDict()
    for element in dictionary:
        dict_r[dictionary[element]] = element
    dict_r[1] = 'UNK'
    dict_r[0] = 'EOS'

    sent = sent.cpu().numpy()
    sent_str = []
    for i in range(len(sent)):
        sent_str += [dict_r[sent[i]]]

    print 'a sentence:',  ' '.join(sent_str)


def segment(sents, dictionary, z_1, z_2, f):
    batch_size = sents.size(0)
    sents = sents.cpu().numpy()
    z_1 = z_1.cpu().numpy()
    z_2 = z_2.cpu().numpy()
    # print z_2
    for i in range(batch_size):
        sent = sents[i, :]
        seg1 = z_1[i, :, :]
        seg2 = z_2[i, :, :]
        sent_str = []
        for j in range(len(sent)):
            if seg1[j, 0] == 1:
                sent_str += ['||']
            if seg2[j, 0] == 1:
                sent_str += ['|||']
            sent_str += [dictionary.idx2word[sent[j]]]

        f.writelines(' '.join(sent_str))


def evaluatePTB(data_source, model, model_params, dictionary):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(model_params['batch_size'])
    f = open('segmentation', 'w')
    it = 0
    for i in range(0, data_source.size(0) - 1, model_params['maxlen']):
        inputs, target = get_batch(data_source, i, model_params['maxlen'])
        loss, hidden, z_1, z_2 = model(inputs, target, hidden)
        segment(inputs, dictionary, z_1.data, z_2.data, f)
        total_loss += loss.data
        hidden = repackage_hidden(hidden)
        it += 1
    f.close()
    PPL = torch.exp(total_loss/it)
    model.train()
    return PPL.cpu().numpy()[0]


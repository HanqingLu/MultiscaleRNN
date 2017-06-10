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
#     return x > 0.5


class bound(Function):
    def forward(self, x):
        # forward : x -> output
        self.save_for_backward(x)
        output = x > 0.5
        return output

    def backward(self, output_grad):
        # backward: output_grad -> x_grad
        x = self.to_save[0]
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



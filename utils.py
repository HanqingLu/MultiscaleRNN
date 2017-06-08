from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import functional as F
from torch.nn import Module
import time


def hard_sigm(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output


def bound(x):
    return x > 0.5


class masked_NLLLoss(Module):
    def __init__(self):
        super(masked_NLLLoss, self).__init__()

    def forward(self, cost, inputs, mask):
        # inputs.size = mask.size = batch_size * 1
        # cost.size = batch_size * dict_size

        loss = Variable(torch.zeros(inputs.size(0))).cuda()
        # cost_flat = cost.flatten()
        # inputs_flat = inputs.flatten()
        # inputs_flat_idx = Variable(torch.arange(inputs.size(0))*cost.size(1)+)
        for i in range(inputs.size(0)):
            loss[i] = - cost[i, inputs[i]] * mask[i]
        # print "NLLLoss time", time.time() - start_time
        return loss

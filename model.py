import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as Func
import torch.optim as optim
from torch.autograd import Variable
import torch
from layers import HM_LSTM
from utils import masked_NLLLoss
import time


class HM_Net(Module):
    def __init__(self, a, size_list, dict_size, embed_size):
        super(HM_Net, self).__init__()
        self.dict_size = dict_size
        self.embed_in = nn.Embedding(dict_size, embed_size)
        self.HM_LSTM = HM_LSTM(a, embed_size, size_list)
        # self.weight_1 = Variable(torch.randn(1, size_list[0]+size_list[1]), require_grads=True)
        # self.weight_2 = Variable(torch.randn(1, size_list[0]+size_list[1]), require_grads=True)
        self.weight = nn.Linear(size_list[0]+size_list[1], 2, bias=False)
        self.embed_out1 = nn.Linear(size_list[0], dict_size, bias=False)
        self.embed_out2 = nn.Linear(size_list[1], dict_size,  bias=False)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax()
        self.loss = masked_NLLLoss()

    def forward(self, inputs, mask):
        # inputs : batch_size * time_steps
        # mask : batch_size * time_steps

        emb = self.embed_in(Variable(inputs))  # batch_size * time_steps * embed_size
        outputs = self.HM_LSTM(emb)  # batch_size *
        time_steps = inputs.size(1)
        batch_size = inputs.size(0)
        mask = Variable(mask, requires_grad=False)
        batch_loss = Variable(torch.zeros(batch_size).cuda())

        for i in range(time_steps):
            h_1 = outputs[4 * i].t()  # batch_size * size_list[0]
            h_2 = outputs[4 * i + 1].t()  #
            h = torch.cat((h_1, h_2), 1)  # batch_size * size_list[0] + size_list[1]

            g = Func.sigmoid(self.weight(h))  # batch_size * 2

            g_1 = g[:, 0:1]  # batch_size * 1
            g_2 = g[:, 1:2]  # batch_size * 1

            h_e1 = g_1.expand(batch_size, self.dict_size)*self.embed_out1(h_1)
            h_e2 = g_2.expand(batch_size, self.dict_size)*self.embed_out2(h_2)

            h_e = self.relu(h_e1 + h_e2)

            word = self.logsoftmax(h_e)  # batch_size * dict_size

            batch_loss += self.loss(word, inputs[:, i], mask[:, i])  # batch_size * 1

        return batch_loss

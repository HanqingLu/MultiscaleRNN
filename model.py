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
        self.size_list = size_list
        self.drop = nn.Dropout(p=0.5)
        self.embed_in = nn.Embedding(dict_size, embed_size)
        self.HM_LSTM = HM_LSTM(a, embed_size, size_list)
        self.weight = nn.Linear(size_list[0]+size_list[1], 2)
        self.embed_out1 = nn.Linear(size_list[0], dict_size)
        self.embed_out2 = nn.Linear(size_list[1], dict_size)
        self.relu = nn.ReLU()
        # self.logsoftmax = nn.LogSoftmax()
        # self.loss = masked_NLLLoss()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, target, hidden):
        # inputs : batch_size * time_steps
        # mask : batch_size * time_steps

        emb = self.embed_in(Variable(inputs, volatile=not self.training))  # batch_size * time_steps * embed_size
        emb = self.drop(emb)
        h_1, h_2, z_1, z_2, hidden = self.HM_LSTM(emb, hidden)  # batch_size * time_steps * hidden_size

        # mask = Variable(mask, requires_grad=False)
        # batch_loss = Variable(torch.zeros(batch_size).cuda())

        h_1 = self.drop(h_1)  # batch_size * time_steps * hidden_size
        h_2 = self.drop(h_2)
        h = torch.cat((h_1, h_2), 2)

        g = Func.sigmoid(self.weight(h.view(h.size(0)*h.size(1), h.size(2))))
        g_1 = g[:, 0:1]  # batch_size * time_steps, 1
        g_2 = g[:, 1:2]

        h_e1 = g_1.expand(g_1.size(0), self.dict_size)*self.embed_out1(h_1.view(h_1.size(0)*h_1.size(1), h_2.size(2)))
        h_e2 = g_2.expand(g_2.size(0), self.dict_size)*self.embed_out2(h_2.view(h_2.size(0)*h_2.size(1), h_2.size(2)))

        h_e = self.relu(h_e1 + h_e2)  # batch_size*time_steps, hidden_size
        batch_loss = self.loss(h_e, Variable(target))

        return batch_loss, hidden, z_1, z_2

    def init_hidden(self, batch_size):
        h_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
        c_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
        z_t1 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)
        h_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
        c_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
        z_t2 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return hidden


class Tutorial_Net(Module):

    def __init__(self, hidden_size, dict_size, embed_size):
        super(Tutorial_Net, self).__init__()
        self.dict_size = dict_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=0.5)
        self.embed_in = nn.Embedding(dict_size, embed_size)
        self.LSTM = nn.LSTM(embed_size, hidden_size, 2, batch_first=True, dropout=0.5)
        self.embed_out = nn.Linear(hidden_size, dict_size)
        self.loss = nn.CrossEntropyLoss()

        self.init_weight()

    def forward(self, inputs, target, hidden):
        emb = self.embed_in(Variable(inputs, volatile=not self.training))  # batch_size * time_steps * embed_size
        emb = self.drop(emb)
        h, hidden = self.LSTM(emb, hidden)
        h = self.drop(h)  # batch_size * time_steps * hidden_size
        h_e = self.embed_out(h.view(h.size(0) * h.size(1), h.size(2)))  # batch_size*time_steps, hidden_size
        batch_loss = self.loss(h_e, Variable(target))
        return batch_loss, hidden

    def init_weight(self):
        initrange = 0.1
        self.embed_in.weight.data.uniform_(-initrange, initrange)
        self.embed_out.bias.data.fill_(0)
        self.embed_out.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(2, batch_size, 650).float().cuda())
        c = Variable(torch.zeros(2, batch_size, 650).float().cuda())

        hidden = (h, c)
        return hidden






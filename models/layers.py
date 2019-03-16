import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features, drop_rate, pooling):
        super(_DenseLayer, self).__init__()
        # self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
        #                 growth_rate, kernel_size=1, stride=1, bias=False)),
        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                 kernel_size=3, stride=1, padding=1, bias=False)),
        self.conv1 = nn.Conv1d(num_input_features, num_output_features, kernel_size=3, stride=1, padding=2, dilation=1)
        self.bn1 = nn.BatchNorm1d(num_output_features)
        self.relu1 = nn.SELU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(x)
        new_features = self.bn1(new_features)
        new_features = self.relu1(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], dim=1)

class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
        )
    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x

# class CatBnAct(nn.Module):
#     def __init__(self, in_chs, activation_fn=nn.SELU(inplace=True)):
#         super(CatBnAct, self).__init__()
#         self.bn = nn.BatchNorm1d(in_chs, eps=0.001)
#         self.act = activation_fn

#     def forward(self, x):
#         x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
#         return self.act(self.bn(x))

class Conv1dBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.SELU(inplace=True)):
        super(BnActConv1d, self).__init__()
        self.bn = nn.BatchNorm1d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv1d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))

class DualPathBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False
        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv1d(
                    in_chs=in_chs, out_chs=num_1x1_c + inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv1d(
                    in_chs=in_chs, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

        self.c1x1_a = BnActConv1d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv1d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv1d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv1d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv1d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :]
            x_s2 = x_s[:, self.num_1x1_c:, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :]
            out2 = x_in[:, self.num_1x1_c:,:]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class luong_gate_attention(nn.Module):
    
    def __init__(self, hidden_size, emb_size, prob=0.2):
        super(luong_gate_attention, self).__init__()
        self.hidden_size, self.emb_size = hidden_size, emb_size
        self.linear_in_conv = nn.Linear(hidden_size, hidden_size)
        self.linear_out_conv = nn.Linear(2*hidden_size, hidden_size)
        self.selu_out_conv = nn.Sequential(nn.SELU(inplace=True))
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)
        self.selu_out = nn.Sequential(nn.SELU(inplace=True))

    def forward(self, h, conv, context):
        """
            Inputs:
                h: bs * ts * hs
                conv: bs * ws * hs ## 短语
                context: bs * ws * hs
        """
        gamma_h = self.linear_in_conv(h) 
        weights = torch.bmm(gamma_h, conv.transpose(1, 2)) ## bs * ts * ws
        weights = F.softmax(weights, dim=-1)

        c_t_conv = torch.bmm(weights, conv) ## bs * ts * hs
        output_conv = self.selu_out_conv(self.linear_out_conv(torch.cat((h, c_t_conv), dim=-1)))

        gamma_h = self.linear_in(output_conv)
        weights = torch.bmm(gamma_h, context.transpose(1, 2)) ## bs * ts * ws
        weights = F.softmax(weights, dim=-1)
        c_t = torch.bmm(weights, context)

        output = self.selu_out(self.linear_out(torch.cat([output_conv, c_t], -1)))

        output = output_conv + output

        return output, weights


class GraphConvolution(nn.Module):

    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


if __name__ == '__main__':
    dual = DualPathBlock(300, 32, 32, 32, )
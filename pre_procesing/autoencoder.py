import math
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ACTIVATION = nn.ReLU()


class GeneratorModulation(torch.nn.Module):
    def __init__(self, styledim, outch):
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        return x * (1 * self.scale(style)) + self.bias(style)


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


def init_weights(m):
    """ initialize weights of fully connected layer
    """
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        # nn.init.constant_(m.bias, 0.01)

    elif type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=30,
                 drop_prob=0.1,
                 code_snape=25,
                 hidden_layers=5,
                 norm_layer=nn.BatchNorm1d,
                 activation=ACTIVATION):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.norm_layer = norm_layer
        self.activation = activation
        self.hidden_layers_list = []
        use_bias = norm_layer != nn.BatchNorm1d
        self.hidden_layers = hidden_layers
        # Layers
        self.input = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=use_bias),
            self.norm_layer(self.hidden_dim),
            self.activation,
            nn.Dropout(self.drop_prob)
        )
        self.input.apply(init_weights)
        for i in range(self.hidden_layers):
            layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=use_bias),
                self.norm_layer(self.hidden_dim),
                self.activation,
                nn.Dropout(drop_prob)
            )
            layer.apply(init_weights)
            self.hidden_layers_list.append(layer)

        self.code = nn.Sequential(
            nn.Linear(self.hidden_dim, code_snape)
        )
        self.code.apply(init_weights)

    def forward(self, x):
        """
        Standard forward
        """
        x = self.input(x)

        for hidden_layer in self.hidden_layers_list:
            x = x + hidden_layer(x)

        return self.code(x)

    def save(self, path):
        state_dict = self.state_dict()
        weights_path = os.path.join(path, "encoder.pt")
        torch.save(state_dict, weights_path)


class Decoder(nn.Module):
    def __init__(self,
                 code_dim,
                 hidden_dim=20,
                 output_dim=25,
                 drop_prob=0.1,
                 hidden_layers=10,
                 norm_layer=nn.BatchNorm1d,
                 activation=ACTIVATION,
                 ):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.norm_layer = norm_layer
        self.activation = activation
        self.hidden_layers = hidden_layers
        self.hidden_layers_list = []
        use_bias = norm_layer != nn.BatchNorm1d
        # Layers
        self.input = nn.Sequential(
            nn.Linear(code_dim, self.hidden_dim),
            self.norm_layer(self.hidden_dim),
            self.activation,
            nn.Dropout(self.drop_prob)
        )
        self.input.apply(init_weights)
        for i in range(self.hidden_layers):
            layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=use_bias),
                self.norm_layer(self.hidden_dim),
                self.activation,
                nn.Dropout(drop_prob)
            )
            layer.apply(init_weights)
            self.hidden_layers_list.append(layer)

        self.last_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.last_layer.apply(init_weights)

    def forward(self, x):
        """
        Concats bio and batch codes, and does forward propagation to obtain recon
        """
        x = self.input(x)

        for hidden_layer in self.hidden_layers_list:
            x = x + hidden_layer(x)

        return self.last_layer(x)

import torch
from torch import nn


def init_weights(m):
    """ initialize weights of fully connected layer
    """
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight, gain=1)
        # m.bias.data.zero_()
    elif type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Net(nn.Module):
    def __init__(self, input_dim=25,
                 hidden_dim=20,
                 drop_prob=0.5,
                 hidden_layers=5,
                 norm_layer=nn.BatchNorm1d,
                 activation=nn.ReLU(),class_num=2):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.norm_layer = norm_layer
        self.activation = activation
        self.hidden_layers = hidden_layers

        use_bias = norm_layer != nn.BatchNorm1d
        self.hidden_layers_list = []

        # Layers
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=use_bias),
            self.norm_layer(self.hidden_dim),
            self.activation,
        )
        self.input_layer.apply(init_weights)
        for i in range(self.hidden_layers):
            layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=use_bias),
                self.norm_layer(self.hidden_dim),
                self.activation,
                nn.Dropout(drop_prob)
            )
            layer.apply(init_weights)
            self.hidden_layers_list.append(layer)
        self.hidden_layers_list = nn.ModuleList(self.hidden_layers_list)

        self.features = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=use_bias),
            self.norm_layer(self.hidden_dim),
            self.activation,
            nn.Dropout(drop_prob)
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Concat bio and batch codes, and does forward propagation to obtain recon
        """
        x = self.input_layer(x)

        for hidden_layer in self.hidden_layers_list:
            x = x + hidden_layer(x)

        activations = self.features(x)
        prediction = self.head(activations).squeeze()

        return prediction, activations


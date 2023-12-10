import torch
import torch.nn as nn

from pre_procesing.autoencoder import Encoder, Decoder


class AutoencoderShrink(nn.Module):
    def __init__(self, input_dim, ae_encoding_dim, hidden_layers):
        super(AutoencoderShrink, self).__init__()
        self.encoder = Encoder(input_dim,
                               hidden_dim=20,
                               drop_prob=0.1,
                               hidden_layers=hidden_layers,
                               code_snape=ae_encoding_dim)

        self.decoder = Decoder(ae_encoding_dim,
                               hidden_dim=20,
                               hidden_layers=hidden_layers,
                               output_dim=input_dim,
                               drop_prob=0.1, )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def from_pretrain(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

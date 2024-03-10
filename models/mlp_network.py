from models.mlp_base import MLPBase
import torch
import os
import numpy as np


class MLPNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, activations, name):
        super(MLPNetwork, self).__init__()
        self.model = MLPBase(input_dim, output_dim, hidden_size, activations)
        self.device = torch.device('cpu')
        self.name = name

    def meta_forward(self, x):
        return self.model.forward(x)

    def forward(self, **kwargs):
        raise NotImplementedError('function forward has not been implemented')

    def weights(self):
        return self.state_dict()

    def load_weights(self, weights):
        self.load_state_dict(weights)

    def save(self, path, version=0):
        self.model.save(os.path.join(path, '{}-{}.pt'.format(self.name, version)))

    def load(self, path, version=0, **kwargs):
        self.model.load(os.path.join(path, '{}-{}.pt'.format(self.name, version)), **kwargs)

    def to(self, device):
        if not device == self.device:
            self.device = device
            super().to(device)

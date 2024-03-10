import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay
        self.device = torch.device('cpu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        elif len(x.shape) == 3:
            if x.shape[0] == weight.data.shape[0]:
                x = torch.einsum('bij,bjk->bik', x, weight)
            else:
                x = torch.einsum('cij,bjk->bcik', x, weight)
        elif len(x.shape) == 4:
            if x.shape[0] == weight.data.shape[0]:
                x = torch.einsum('cbij,cjk->cbik', x, weight)
            else:
                x = torch.einsum('cdij,bjk->bcdik', x, weight)
        elif len(x.shape) == 5:
            x = torch.einsum('bcdij,bjk->bcdik', x, weight)
        assert x.shape[0] == bias.shape[0] and x.shape[-1] == bias.shape[-1]

        if len(x.shape) == 4:
            bias = bias.unsqueeze(1)
        elif len(x.shape) == 5:
            bias = bias.unsqueeze(1)
            bias = bias.unsqueeze(1)

        x = x + bias

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss
    def to(self, device):
        if not device == self.device:
            self.device = device
            super().to(device)
            self.weight = self.weight.to(self.device)
            self.bias = self.bias.to(self.device)
            self.saved_weight = self.saved_weight.to(self.device)
            self.saved_bias = self.saved_bias.to(self.device)

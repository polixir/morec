import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rnn_base import RNNBase

class EnsembleMLPBase(RNNBase):
    def __init__(self, input_size, output_size, ensemble_size, hidden_size_list, activation):
        super().__init__(input_size, output_size, hidden_size_list, activation, [f'efc-{ensemble_size}'] * len(activation))

    def meta_forward(self, x, h=None, require_full_hidden=False):
        return super(EnsembleMLPBase, self).meta_forward(x, [], False)[0]

    def forward(self, *args, **kwargs):
        return self.meta_forward(*args, **kwargs)

if __name__ == '__main__':
    hidden_layers = [256, 128, 64]
    hidden_activates = ['leaky_relu'] * len(hidden_layers)
    hidden_activates.append('tanh')
    nn = EnsembleMLPBase(64, 4, 3, hidden_layers, hidden_activates)
    for _ in range(5):
        x = torch.randn((3, 64))
        y = nn.forward(x)
        print(y)
        print(x.shape, y.shape)

import torch.types

from models.ensemble_mlp_network import EnsembleMLPNetwork as MLPNetwork

class EnsembleEncoderBase(torch.nn.Module):
    def __init__(self):
        super(EnsembleEncoderBase, self).__init__()
        self.device = torch.device('cpu')

class EnsembleEncoderEmpty(EnsembleEncoderBase):
    def __init__(self):
        super(EnsembleEncoderEmpty, self).__init__()

    def parameters(self, recursive=True):
        return []

    def encode_state(self, state):
        return state

    def encode(self, state, action):
        return torch.cat((state, action), dim=-1)

    def to(self, device):
        return

    def save(self, path, version):
        return

    def load(self, path, version, **kwargs):
        return

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return dict()

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        return


class EnsembleEncoder(EnsembleEncoderBase):
    def __init__(self, obs_dim, act_dim, ensemble_size,
                 state_embedding_size, action_embedding_size,
                 state_embedding_hidden_size,
                 state_embedding_activations, action_embedding_hidden_size,
                 action_embedding_activations, name_suffix=''):
        super(EnsembleEncoder, self).__init__()
        self.name_suffix = name_suffix
        self.state_encoder = MLPNetwork(obs_dim, state_embedding_size, ensemble_size, state_embedding_hidden_size,
                                        state_embedding_activations, 'state_encoder'+name_suffix)
        self.action_embedding_size = action_embedding_size
        self.action_encoder = MLPNetwork(act_dim, action_embedding_size, ensemble_size, action_embedding_hidden_size,
                                         action_embedding_activations, 'action_encoder'+name_suffix)

    def parameters(self, recursive=True):
        return [*self.state_encoder.parameters(True)] + [*self.action_encoder.parameters(True)]

    def split_encode(self, state, action):
        return self.state_encoder.meta_forward(state), self.action_encoder.meta_forward(action)

    def encode(self, state, action):
        action_mask = 1.0 if self.action_embedding_size > 1 else 0.0
        return torch.cat((self.state_encoder.meta_forward(state), self.action_encoder.meta_forward(action) * action_mask), dim=-1)

    def to(self, device):
        if not self.device == device:
            self.device = device
            self.state_encoder.to(device)
            self.action_encoder.to(device)

    def save(self, path, version):
        self.state_encoder.save(path, version)
        self.action_encoder.save(path, version)

    def load(self, path, version, **kwargs):
        self.state_encoder.load(path, version, **kwargs)
        self.action_encoder.load(path, version, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return dict(
            state_encoder=self.state_encoder.state_dict(destination, prefix, keep_vars),
            action_encoder=self.action_encoder.state_dict(destination, prefix, keep_vars)
        )

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        if 'state_encoder' in state_dict:
            self.state_encoder.load_state_dict(state_dict['state_encoder'], strict)
        else:
            print(f'state encoder is not founded in the state dict')
        if 'action_encoder' in state_dict:
            self.action_encoder.load_state_dict(state_dict['action_encoder'], strict)
        else:
            print(f'action encoder is not founded in the state dict')


class EnsembleEncoderVanilla(EnsembleEncoderBase):
    def __init__(self, obs_dim, act_dim, ensemble_size, embedding_size, hidden_size, hidden_activations, name_suffix=''):
        super(EnsembleEncoderVanilla, self).__init__()

        self.encoder = MLPNetwork(obs_dim + act_dim, embedding_size, ensemble_size, hidden_size,
                                        hidden_activations, 'vanilla_encoder' + name_suffix)

    def parameters(self, recursive=True):
        return [*self.encoder.parameters(True)]

    def encode(self, state, action):
        return self.encoder.meta_forward(torch.cat((state, action), dim=-1))

    def to(self, device):
        if not self.device == device:
            self.device = device
            self.encoder.to(device)

    def save(self, path, version):
        self.encoder.save(path, version)

    def load(self, path, version, **kwargs):
        self.encoder.load(path, version, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return dict(
            vanilla_encoder=self.encoder.state_dict(destination, prefix, keep_vars),
        )

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        if 'vanilla_encoder' in state_dict:
            self.encoder.load_state_dict(state_dict['vanilla_encoder'], strict)
        else:
            print(f'vanilla encoder is not founded in the state dict')


class EnsembleEncoderVDB(EnsembleEncoderBase):
    def __init__(self, obs_dim, act_dim, ensemble_size, embedding_size, hidden_size, hidden_activations, name_suffix=''):
        super(EnsembleEncoderVDB, self).__init__()
        self.fc = MLPNetwork(obs_dim + act_dim, hidden_size[-1], ensemble_size, hidden_size[:-1],
                             hidden_activations[:-1], 'VDB_encoder_fc' + name_suffix)
        self.mu = MLPNetwork(hidden_size[-1], embedding_size, ensemble_size, [],
                             hidden_activations[-1:], 'VDB_encoder_mu' + name_suffix)
        self.logvar = MLPNetwork(hidden_size[-1], embedding_size, ensemble_size, [],
                                hidden_activations[-1:], 'VDB_encoder_logvar' + name_suffix)

    def parameters(self, recursive=True):
        return [*self.fc.parameters(True)] + [*self.mu.parameters(True)] + [*self.logvar.parameters(True)]

    def encode_z(self, state, action):
        x = self.fc.meta_forward(torch.cat((state, action), dim=-1))
        mu = self.mu.meta_forward(x)
        logvar = self.logvar.meta_forward(x)
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + std * eps, mu, logvar

    def encode(self, state, action):
        x = self.fc.meta_forward(torch.cat((state, action), dim=-1))
        mu = self.mu.meta_forward(x)
        return mu

    def to(self, device):
        if not self.device == device:
            self.device = device
            self.fc.to(device)
            self.mu.to(device)
            self.logvar.to(device)

    def save(self, path, version):
        self.fc.save(path, version)
        self.mu.save(path, version)
        self.logvar.save(path, version)

    def load(self, path, version, **kwargs):
        self.fc.load(path, version, **kwargs)
        self.mu.load(path, version, **kwargs)
        self.logvar.load(path, version, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return dict(
            encoder_fc=self.fc.state_dict(destination, prefix, keep_vars),
            encoder_mu=self.mu.state_dict(destination, prefix, keep_vars),
            encoder_logvar=self.logvar.state_dict(destination, prefix, keep_vars)
        )

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        if 'encoder_fc' in state_dict:
            self.fc.load_state_dict(state_dict['encoder_fc'], strict)
        else:
            print(f'encoder fc is not founded in the state dict')
        if 'encoder_mu' in state_dict:
            self.mu.load_state_dict(state_dict['encoder_mu'], strict)
        else:
            print(f'encoder mu is not founded in the state dict')
        if 'encoder_logvar' in state_dict:
            self.logvar.load_state_dict(state_dict['encoder_logvar'], strict)
        else:
            print(f'encoder logvar is not founded in the state dict')
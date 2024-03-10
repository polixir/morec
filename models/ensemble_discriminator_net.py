import torch.types

from models.ensemble_mlp_network import EnsembleMLPNetwork as MLPNetwork
class EnsembleDiscrNetBase:
    def __init__(self):
        self.device = torch.device('cpu')
    def parameters(self, recursive):
        pass
    def run_logits(self, state, action):
        pass
    def to(self, device):
        pass
    def save(self, path, version):
        pass
    def load(self, path, version, **kwargs):
        pass

    def state_dict(self):
        pass
    def load_state_dict(self, state_dict):
        pass
    
class EnsembleGAILDiscrNet(EnsembleDiscrNetBase):
    def __init__(self, input_size, ensemble_size, hidden_size,
                 activations, name_suffix=''):
        super(EnsembleGAILDiscrNet, self).__init__()
        self.name_suffix = name_suffix
        self.model = MLPNetwork(input_size, 1, ensemble_size, hidden_size, activations, name_suffix)
        
    def parameters(self, recursive=True):
        return [*self.model.parameters(True)]
    
    def run_logits(self, x):
        return self.model.meta_forward(x)
    
    def to(self, device):
        if not self.device == device:
            self.device = device
            self.model.to(device)
    
    def save(self, path, version):
        self.model.save(path, version)
        
    def load(self, path, version, **kwargs):
        self.model.load(path, version, **kwargs)
        
    def state_dict(self):
        return dict(
            discriminator=self.model.state_dict()
        )
    
    def load_state_dict(self, state_dict):
        if 'discriminator' in state_dict:
            self.model.load_state_dict(state_dict['discriminator'])
        else:
            print(f'discriminator is not founded in the state dict')

        
class EnsembleAIRLDiscrNet(EnsembleDiscrNetBase):
    def __init__(self, input_size_r, input_size_V, ensemble_size, hidden_size_r, hidden_size_V, activations_r, activations_V,
                 gamma, name_suffix=''):
        super(EnsembleAIRLDiscrNet, self).__init__()
        self.name_suffix_r = name_suffix + '_r'
        self.name_suffix_V = name_suffix + '_V'
        self.model_r = MLPNetwork(input_size_r, 1, ensemble_size, hidden_size_r, activations_r, self.name_suffix_r)
        self.model_V = MLPNetwork(input_size_V, 1, ensemble_size, hidden_size_V, activations_V, self.name_suffix_V)
        self.gamma = gamma
        
    def parameters(self, recursive=True):
        return [*self.model_r.parameters(True)] + [*self.model_V.parameters(True)]
        
    def run_logits(self, embed_for_r, embed_for_V, next_embed_for_V, dones, log_pi_s):
        r_s = self.model_r.meta_forward(embed_for_r)
        V_s = self.model_V.meta_forward(embed_for_V)
        V_s_next = self.model_V.meta_forward(next_embed_for_V)
        f = r_s + self.gamma * (1 - dones) * V_s_next - V_s
        return f - log_pi_s

    def to(self, device):
        if not self.device == device:
            self.device = device
            self.model_r.to(device)
            self.model_V.to(device)

    def save(self, path, version):
        self.model_r.save(path, version)
        self.model_V.save(path, version)
    
    def load(self, path, version, **kwargs):
        self.model_r.load(path, version, **kwargs)
        self.model_V.load(path, version, **kwargs)

    def state_dict(self):
        return dict(
            discriminator_r=self.model_r.state_dict(),
            discriminator_V=self.model_V.state_dict(),
        )

    def load_state_dict(self, state_dict):
        if 'discriminator_r' in state_dict:
            self.model_r.load_state_dict(state_dict['discriminator_r'])
        else:
            print(f'discriminator_r is not founded in the state dict')
        if 'discriminator_V' in state_dict:
            self.model_V.load_state_dict(state_dict['discriminator_V'])
        else:
            print(f'discriminator_V is not founded in the state dict')

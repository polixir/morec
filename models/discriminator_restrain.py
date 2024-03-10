from models.discriminator_net import GAILDiscrNet, AIRLDiscrNet
import torch
import numpy as np
import os
from .DataLoader import DataLoader
from collections import deque
import copy
from models.encoder import EncoderEmpty, EncoderVanilla, Encoder, EncoderVDB

class DiscriminatorRestrain:
    def __init__(self, obs_dim, z_dim, act_dim, next_obs_dim, gamma, state_embedding_size, action_embedding_size, state_embedding_hidden_size,
                 state_embedding_activations, action_embedding_hidden_size, action_embedding_activations,
                 discriminator_activations, discriminator_hidden_size, transition_hidden,
                 transition_activations, expert_path, expert_traj_num, discriminator_lr, transition_lr,
                 history_discriminator_interval, max_maintaining_disc, mi_regulizer_factor,
                 encoder_type="restrain", vanilla_encoder_activations=["relu", "relu"],
                 vanilla_encoder_hidden_size=[128], vanilla_encoder_embedding_size=256,
                 use_AIRL=False, use_state_only_reward=False, transition_learner_data=False, target_transition_loss_ratio=-1.0,
                 no_regret_discriminator=False, regret_regulizer_factor=0.0, auto_ent_coeff_optim=False,
                 output_clip_value=None, delta_input=True, d_2nd_loss_type='none'):
        if use_state_only_reward:
            assert(use_AIRL)
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.z_dim = z_dim
        self.output_clip_value = output_clip_value
        self.d_2nd_loss_type = d_2nd_loss_type
        self.next_obs_dim = next_obs_dim
        self.transition_learner_data = transition_learner_data
        self.state_embedding_size = state_embedding_size
        self.action_embedding_size = action_embedding_size
        self.no_regret_discriminator = no_regret_discriminator
        self.regret_regulizer_factor = regret_regulizer_factor
        self.encoder_type = encoder_type
        self.state_embedding_hidden_size = state_embedding_hidden_size
        self.state_embedding_activations = state_embedding_activations
        self.action_embedding_hidden_size = action_embedding_hidden_size
        self.action_embedding_activations = action_embedding_activations
        self.discriminator_hidden_size = discriminator_hidden_size
        self.discriminator_activations = discriminator_activations
        self.vanilla_encoder_embedding_size = vanilla_encoder_embedding_size
        self.vanilla_encoder_hidden_size = vanilla_encoder_hidden_size
        self.vanilla_encoder_activations = vanilla_encoder_activations
        if encoder_type == 'restrain':
            self.encoder_input_dim = obs_dim + z_dim + act_dim
            self.encoder = Encoder(
                self.encoder_input_dim, obs_dim, state_embedding_size, action_embedding_size, state_embedding_hidden_size,
                state_embedding_activations, action_embedding_hidden_size, action_embedding_activations, name_suffix=""
            )
            self.discriminator_input_size = state_embedding_size + action_embedding_size
            self.encoder_target = Encoder(
                self.encoder_input_dim, state_embedding_size, action_embedding_size, state_embedding_hidden_size,
                state_embedding_activations, action_embedding_hidden_size, action_embedding_activations,
                name_suffix="_target"
            )
        else:
            self.encoder_input_dim = obs_dim + z_dim + act_dim
            self.encoder = EncoderVanilla(
                self.encoder_input_dim, obs_dim, vanilla_encoder_embedding_size, vanilla_encoder_hidden_size, vanilla_encoder_activations
            )
            self.discriminator_input_size = vanilla_encoder_embedding_size
            self.encoder_target = EncoderVanilla(
                self.encoder_input_dim, obs_dim, vanilla_encoder_embedding_size, vanilla_encoder_hidden_size,
                vanilla_encoder_activations
            )
        self.discriminator = GAILDiscrNet(self.discriminator_input_size, discriminator_hidden_size, discriminator_activations, 'discriminator')



        # target network is used to calculate ensemble rewards
        self.discriminator_target = GAILDiscrNet(self.discriminator_input_size, discriminator_hidden_size,
                                    discriminator_activations, 'discriminator_target')
        self.spectrum_norm = False
        if self.spectrum_norm:
            self.encoder.encoder.model.set_all_layer_spectral_norm()
            self.encoder_target.encoder.model.set_all_layer_spectral_norm()
            self.discriminator.model.model.set_all_layer_spectral_norm()
            self.discriminator_target.model.model.set_all_layer_spectral_norm()

        self.discrim_parameter = [*self.encoder.parameters(True)] + \
                                 [*self.discriminator.parameters(True)]

        self.expert_data = DataLoader(expert_path)
        self.device = torch.device('cpu')
        self.version = 0
        self.update_time = 0
        self.discriminator_lr = discriminator_lr
        self.discriminator_optimizer = torch.optim.Adam(self.discrim_parameter, lr=discriminator_lr)
        self.history_discriminator_interval = history_discriminator_interval
        self.history_discriminator = deque(maxlen=max_maintaining_disc)
        self.middle_value = 0.0
        self.expert_value = 1.0
        self.learner_value = 0.0
        self.middle_value_backup = 0.0
        self.expert_value_backup = 1.0
        self.learner_value_backup = 0.0
        self.max_expert_value = 1.0
        self.minimum_learner_value = 0.0
        self.middle_value_buffer = []
        self.mi_regulizer_factor = mi_regulizer_factor
        self.target_transition_loss_ratio = target_transition_loss_ratio
        self.learnable_mi_factor = self.target_transition_loss_ratio > 0.0

        self.log_eta = (torch.ones((1)).to(torch.get_default_dtype()) *
                              np.log(0.2)).to(self.device).requires_grad_(True)
        self.eta_optimizer = torch.optim.Adam([self.log_eta], lr=3e-2)


        self.ent_coeff = 0.001
        self.log_ent_coeff = (torch.ones((1)).to(torch.get_default_dtype()) *
                              np.log(self.ent_coeff)).to(self.device).requires_grad_(True)
        self.ent_coeff_optimizer = torch.optim.Adam([self.log_ent_coeff], lr=1e-2)
        self.auto_ent_coeff_optim = auto_ent_coeff_optim
        self.target_ent = 0.35
        self.delta_input = delta_input

        self.use_AIRL = use_AIRL
        self.use_state_only_reward = use_state_only_reward
        if no_regret_discriminator:
            self.load_dict_target_weight(self.state_dict())

    def to(self, device):
        if not self.device == device:
            self.device = device
            self.encoder.to(self.device)
            self.discriminator.to(self.device)
            self.encoder_target.to(self.device)
            self.discriminator_target.to(self.device)
            self.expert_data.to(self.device)
            self.log_eta = self.log_eta.to(self.device)

    def state_dict(self):
        state_dict = dict()
        state_dict_encoder = self.encoder.state_dict()
        state_dict.update(state_dict_encoder)
        state_dict['discriminator'] = self.discriminator.state_dict()
        state_dict['middle_value'] = self.middle_value_backup
        state_dict['expert_value'] = self.expert_value_backup
        state_dict['learner_value'] = self.learner_value_backup
        self.middle_value_buffer = []
        return copy.deepcopy(state_dict)

    def load_dict_target_weight(self, state_dict):
        self.encoder_target.load_state_dict(state_dict)
        self.discriminator_target.load_state_dict(state_dict['discriminator'])

    def load_dict_target(self, state_dict):
        self.load_dict_target_weight(state_dict)
        if 'middle_value' in state_dict:
            self.middle_value = state_dict['middle_value']
        else:
            print(f'middle value missing')
        if 'expert_value' in state_dict:
            self.expert_value = state_dict['expert_value']
        else:
            print(f'expert value missing')
        if 'learner_value' in state_dict:
            self.learner_value = state_dict['learner_value']
        else:
            print(f'learner value missing')

    def load_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict)
        self.discriminator.load_state_dict(state_dict['discriminator'])

    def save(self, path):
        self.encoder.save(path, version=self.version)
        self.discriminator.save(path, version=self.version)
        torch.save(self.history_discriminator, os.path.join(path, 'historical_data.pt'))

    def load(self, path, version=0, **kwargs):
        self.encoder.load(path, version=version, **kwargs)
        self.discriminator.load(path, version=version, **kwargs)
        self.history_discriminator = torch.load(os.path.join(path, 'historical_data.pt'))
        # self.history_discriminator = [self.history_discriminator[i] for i in range(20, len(self.history_discriminator))]
        if len(self.history_discriminator) > 0:
            self.max_expert_value = -10000.0
            self.minimum_learner_value = 100000.
            for disc in self.history_discriminator:
                if disc['expert_value'] > self.max_expert_value:
                    self.max_expert_value = disc['expert_value']
                if disc['learner_value'] < self.minimum_learner_value:
                    self.minimum_learner_value = disc['learner_value']
            print(f'max and minimum discriminator output: {self.max_expert_value}, {self.minimum_learner_value}')

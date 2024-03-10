from models.discriminator_restrain import DiscriminatorRestrain
from .DataLoader import DataLoader
import json
from offlinerlkit.common_config.load_config import init_smart_logger
import smart_logger
import os
import torch
import numpy as np
from models.ensemble_linear_model import EnsembleLinear
from models.ensemble_discriminator_net import EnsembleGAILDiscrNet
from models.ensemble_encoder import EnsembleEncoder, EnsembleEncoderVanilla
class DiscriminatorEnsemble:
    def __init__(self, discriminator_restrain: DiscriminatorRestrain, ensemble_choosing_interval:int=None, max_retain_num: int=None, load_min_max: bool=True):
        self.discriminator_restrain = discriminator_restrain
        if not load_min_max:
            self.discriminator_restrain.minimum_learner_value = 0.0
            self.discriminator_restrain.max_expert_value = 1.0

        discriminator_list = self.discriminator_restrain.history_discriminator
        discriminator_list = [_ for _ in discriminator_list]
        self.max_retain_num = max_retain_num
        if self.max_retain_num is not None and self.max_retain_num > 0:
            self.max_retain_num = int(self.max_retain_num)

            discriminator_list = discriminator_list[-self.max_retain_num:]
        if ensemble_choosing_interval is None:
            self.ensemble_choosing_interval = discriminator_restrain.history_discriminator_interval
        else:
            self.ensemble_choosing_interval = ensemble_choosing_interval

        if self.ensemble_choosing_interval > 1:
            discriminator_list = discriminator_list[::self.ensemble_choosing_interval]
        self.discriminator_list = discriminator_list
        self.ensemble_size = len(self.discriminator_list)
        self.encoder, self.discriminator, self.mask_tensor, self.expert_value_tensor, self.learner_value_tensor = self.init_ensemble_network()
        self.load_min_max = load_min_max
        if load_min_max:
            self.max_expert_value = self.discriminator_restrain.max_expert_value
            self.minimum_learner_value = self.discriminator_restrain.minimum_learner_value
        else:
            self.max_expert_value = 1.0
            self.minimum_learner_value = 0.0
        self.mask_tensor = torch.nonzero(self.mask_tensor > 0).squeeze(dim=-1)
        self.expert_value_tensor = self.expert_value_tensor[self.mask_tensor]
        self.learner_value_tensor = self.learner_value_tensor[self.mask_tensor]
        self.expert_data = self.discriminator_restrain.expert_data
        self.device = torch.device('cpu')

    def init_ensemble_network(self):
        if self.discriminator_restrain.encoder_type == 'restrain':
            encoder = EnsembleEncoder(
                self.discriminator_restrain.encoder_input_dim, self.discriminator_restrain.obs_dim, self.ensemble_size, self.discriminator_restrain.state_embedding_size, self.discriminator_restrain.action_embedding_size,
                self.discriminator_restrain.state_embedding_hidden_size,
                self.discriminator_restrain.state_embedding_activations, self.discriminator_restrain.action_embedding_hidden_size, self.discriminator_restrain.action_embedding_activations, name_suffix=""
            )
        else:

            encoder = EnsembleEncoderVanilla(
                self.discriminator_restrain.encoder_input_dim, self.discriminator_restrain.obs_dim, self.ensemble_size, self.discriminator_restrain.vanilla_encoder_embedding_size, self.discriminator_restrain.vanilla_encoder_hidden_size,
                self.discriminator_restrain.vanilla_encoder_activations
            )
        discriminator = EnsembleGAILDiscrNet(self.discriminator_restrain.discriminator_input_size, self.ensemble_size, self.discriminator_restrain.discriminator_hidden_size,
                                          self.discriminator_restrain.discriminator_activations, 'discriminator')
        # for layer in discriminator.model.model.layer_list:
        #     print(self.ensemble_size, layer.weight.shape, layer.bias.shape)
        # for layer in encoder.encoder.model.layer_list:
        #     print(self.ensemble_size, layer.weight.shape, layer.bias.shape)

        mask_list = []
        expert_value_list = []
        learner_value_list = []

        with torch.no_grad():
            for ind, state_dict in enumerate(self.discriminator_list):
                self.discriminator_restrain.load_dict_target(state_dict)
                middle_value = self.discriminator_restrain.middle_value
                expert_value = self.discriminator_restrain.expert_value
                learner_value = self.discriminator_restrain.learner_value
                mask_list.append(1 if middle_value is not None else 0)
                expert_value_list.append(expert_value if expert_value is not None else 1)
                learner_value_list.append(learner_value if learner_value is not None else 0)

                for layer_ind, layer in enumerate(self.discriminator_restrain.discriminator_target.model.model.layer_list):
                    discriminator.model.model.layer_list[layer_ind].weight.data[ind, ...].copy_(layer.weight.data.transpose(0, 1))
                    discriminator.model.model.layer_list[layer_ind].bias.data[ind, 0, ...].copy_(layer.bias.data)
                for layer_ind, layer in enumerate(self.discriminator_restrain.encoder_target.encoder.model.layer_list):
                    encoder.encoder.model.layer_list[layer_ind].weight.data[ind, ...].copy_(layer.weight.data.transpose(0, 1))
                    encoder.encoder.model.layer_list[layer_ind].bias.data[ind, 0, ...].copy_(layer.bias.data)

        mask_tensor = torch.Tensor(mask_list).to(torch.get_default_dtype()).to(self.discriminator_restrain.device)
        expert_value_tensor = torch.Tensor(expert_value_list).to(torch.get_default_dtype()).to(self.discriminator_restrain.device)
        learner_value_tensor = torch.Tensor(learner_value_list).to(torch.get_default_dtype()).to(self.discriminator_restrain.device)
        return encoder, discriminator, mask_tensor, expert_value_tensor, learner_value_tensor

    def get_total_historical_reward(self, state, z, action, next_state, aquire_prob=False, clip_reward=True):
        state = self.discriminator_restrain.expert_data.norm_raw_obs(state)
        if self.discriminator_restrain.delta_input:
            ensemble_encoding = self.encoder.encode(torch.cat((state, z, action), dim=-1), next_state - state)
        else:
            ensemble_encoding = self.encoder.encode(torch.cat((state, z, action), dim=-1), next_state)
        logits = self.discriminator.run_logits(ensemble_encoding)
        logits_origin = logits

        logits = logits[self.mask_tensor, ...]
        desire_shape = [logits.shape[0]] + [1] * (len(logits.shape) - 1)

        learner_value_tensor = self.learner_value_tensor.reshape(desire_shape)
        expert_value_tensor = self.expert_value_tensor.reshape(desire_shape)
        if clip_reward:
            logits = torch.clamp(logits, learner_value_tensor, expert_value_tensor)
            rge = 2 * max(np.abs(self.max_expert_value), np.abs(self.minimum_learner_value))
            logits = (logits - learner_value_tensor) / (expert_value_tensor - learner_value_tensor + 1e-7) * rge - rge / 2.0

        if aquire_prob:
            prob = torch.sigmoid(logits)
            r = prob
        else:
            r = logits + torch.nn.functional.softplus(-logits)

        return r

    def to(self, device):
        if not device == self.device:
            self.device = device
            self.discriminator_restrain.to(self.device)
            self.discriminator.to(self.device)
            self.encoder.to(self.device)
            self.expert_value_tensor = self.expert_value_tensor.to(self.device)
            self.learner_value_tensor = self.learner_value_tensor.to(self.device)
            self.mask_tensor = self.mask_tensor.to(self.device)


def reward_strategy_apply(ensemble_strategy, full_fake_reward, clip_value=0.5):
    if ensemble_strategy == 'prob_mean':
        full_fake_reward = -torch.log(
            1 - torch.mean(torch.clamp(full_fake_reward, 0.0, 1.0), dim=-1, keepdim=True) + 1e-9)
    elif ensemble_strategy == 'prob_mean_clip':
        full_fake_reward = -torch.log(
            1 - torch.mean(torch.clamp(full_fake_reward, 0.0, clip_value), dim=-1, keepdim=True) + 1e-9)
    else:
        raise NotImplementedError(f'ensemble strategy has not been implemented!!')
    return full_fake_reward

def merge_full_fake_reward(full_fake_reward, clip_value=0.5, ensemble_strategy='prob_mean_clip'):
    if isinstance(full_fake_reward, list):
        full_fake_reward = torch.cat(full_fake_reward, dim=-1)

    if full_fake_reward.shape[-1] > 1:
        full_fake_reward = reward_strategy_apply(ensemble_strategy, full_fake_reward, clip_value)
    elif ensemble_strategy in ['prob_mean', 'prob_mean_clip']:
        full_fake_reward = -torch.log(1 - full_fake_reward + 1e-9)
    full_fake_reward_tmp = full_fake_reward
    return full_fake_reward_tmp

def historical_transition_reward_iteration(discriminator, state, z, action, next_state, clip_value=1.0, ensemble_strategy='prob_mean_clip', clip_reward=True):
    use_prob = ensemble_strategy in ['prob_mean', 'prob_mean_clip']
    with torch.no_grad():
        reward = discriminator.get_total_historical_reward(state, z, action, next_state, aquire_prob=use_prob, clip_reward=clip_reward)
        reward = merge_full_fake_reward(reward, clip_value=clip_value, ensemble_strategy=ensemble_strategy)
    return reward
def historical_transition_reward(discriminator_ensemble, state, z, action, next_state, clip_value=1.0, ensemble_strategy='prob_mean_clip'):
    use_prob = ensemble_strategy in ['prob_mean', 'prob_mean_clip']
    with torch.no_grad():
        reward = discriminator_ensemble.get_total_historical_reward(state, z, action, next_state, aquire_prob=use_prob, clip_reward=True)
        reward = reward.squeeze(-1)
        reward = reward.transpose(0, 1)
        reward = merge_full_fake_reward(reward, clip_value=clip_value, ensemble_strategy=ensemble_strategy)
    return reward

def historical_transition_reward_multi_batch(discriminator_ensemble, state, z, action, next_state, batch_num=1, clip_value=1.0, ensemble_strategy='prob_mean_clip'):
    use_prob = ensemble_strategy in ['prob_mean', 'prob_mean_clip']
    reward_list = []

    with torch.no_grad():
        if batch_num > state.shape[0]:
            batch_num = state.shape[0]
        batch_size = int(np.ceil(state.shape[0] / batch_num))
        for i in range(batch_num):
            if i * batch_size >= state.shape[0]:
                break
            r = discriminator_ensemble.get_total_historical_reward(state[i*batch_size:(i+1)*batch_size], z[i*batch_size:(i+1)*batch_size], action[i*batch_size:(i+1)*batch_size], next_state[i*batch_size:(i+1)*batch_size], aquire_prob=use_prob, clip_reward=True)
            r = r.squeeze(-1)
            r = r.transpose(0, 1)
            reward_list.append(r)


        reward = torch.cat(reward_list, dim=0)
        reward = merge_full_fake_reward(reward, clip_value=clip_value, ensemble_strategy=ensemble_strategy)
    return reward

def load_discriminator_ensemble(reward_dir_name, dir_name=None):
    if dir_name is not None:
        reward_log_path = os.path.join(dir_name, reward_dir_name)
    else:
        reward_log_path = os.path.join(smart_logger.get_base_path(), 'dynamics_reward_models', reward_dir_name)
    parameter_path = os.path.join(reward_log_path, 'config', 'parameter.json')
    parameter = json.load(open(parameter_path, 'r'))
    data = DataLoader(parameter['expert_file_path'])
    discriminator_output_clip_value = None if parameter['discriminator_output_clip_value'] < 0.0 else parameter[
        'discriminator_output_clip_value']

    discriminator_ensemble = DiscriminatorRestrain(data.obs_dim, data.hidden_dim, data.act_dim,
                                                   data.obs_dim, parameter['gamma'], 64, 16,
                                                   [256, 128], ['relu', 'relu', 'relu'], [256, 128],
                                                   ['relu', 'relu', 'relu'],
                                                   parameter['discriminator_hidden_activations'],
                                                   parameter['discriminator_hidden_size'],
                                                   parameter['discriminator_hidden_size'],
                                                   parameter['discriminator_hidden_activations'],
                                                   parameter['expert_file_path'], 100,
                                                   parameter['discriminator_lr'], 1e-3,
                                                   history_discriminator_interval=parameter[
                                                       'history_discriminator_interval'],
                                                   max_maintaining_disc=parameter['max_maintaining_disc'],
                                                   mi_regulizer_factor=0.0 if parameter['vanilla_gail'] else 0.01,
                                                   encoder_type='vanilla' if parameter['vanilla_gail'] else 'restrain',
                                                   auto_ent_coeff_optim=parameter['auto_ent_coeff_optim'],
                                                   output_clip_value=discriminator_output_clip_value,
                                                   delta_input=parameter['delta_input'],
                                                   d_2nd_loss_type=parameter['d_2nd_loss_type'])
    model_path = os.path.join(reward_log_path, 'model')
    discriminator_ensemble.load(model_path, map_location='cpu')
    return discriminator_ensemble


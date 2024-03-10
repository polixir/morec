import os
import time

import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from smart_logger import Logger as sLogger
from models.discriminator_ensemble_loader import historical_transition_reward, historical_transition_reward_multi_batch
from torch.distributions.categorical import Categorical
import psutil
class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        dynamics_rewards = None,
        dynamics_reward_factor: float = 0.0,
        reward_use_type: str = None,
        resample_num_in_transition: int = 1,
        d_clip: float = 0.999,
        reward_infer_batch_num: int=20,
        terminal_dynamics_reward: float=0.6,
        minimal_rollout_length: int=0,
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self._dynamics_rewards = dynamics_rewards
        self._dynamics_reward_factor = dynamics_reward_factor
        self._reward_use_type = reward_use_type
        self._resample_num_in_transition = resample_num_in_transition
        self._d_clip = d_clip
        self._reward_infer_batch_num = reward_infer_batch_num
        self._terminal_dynamics_reward = terminal_dynamics_reward
        self._lst_input_tensor = None
        self._lst_obs_act_tensor = None
        self._minimal_rollout_length = minimal_rollout_length

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        deterministic: bool = False,
        # clipping_value: int = 1e6,# neorl envs use 1e6 clip
        clipping_value: int = 25, # d4rl envs use 25 clip
        current_step = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))
        # print(f'std_mean: {std.shape} {std.mean(axis=0)} {std.mean()}')
        info = {}

        sampled_rewards = None

        if self._reward_use_type in ['select_elite', 'softmax_elite'] and self._dynamics_rewards is not None:
            elites = self.model.elites.cpu().detach().numpy().reshape((-1,))
            mean_tmp = mean[elites, ...]
            std_tmp = std[elites, ...]
            # mean_tmp = mean
            # std_tmp = std
            if not deterministic:
                if self._resample_num_in_transition > 1:
                    ensemble_sample_list = [(mean_tmp + np.random.normal(size=mean_tmp.shape) * std_tmp).astype(np.float32) for i in range(self._resample_num_in_transition)]
                    ensemble_samples = np.concatenate(ensemble_sample_list, axis=0)
                else:
                    ensemble_samples = (mean_tmp + np.random.normal(size=mean_tmp.shape) * std_tmp).astype(np.float32)
            else:
                ensemble_sample_list = [mean_tmp.astype(np.float32)] + [(mean_tmp + np.random.normal(size=mean_tmp.shape) * std_tmp).astype(np.float32) for i in range(self._resample_num_in_transition)]
                ensemble_samples = np.concatenate(ensemble_sample_list, axis=0)

            ensemble_samples = np.clip(ensemble_samples, -clipping_value, clipping_value)
            # choose one model from ensemble
            num_models, batch_size, _ = ensemble_samples.shape
            obs_norm = self._dynamics_rewards.expert_data.normalize_obs(obs)
            act_norm = self._dynamics_rewards.expert_data.normalize_act(action)
            next_obs_norm = self._dynamics_rewards.expert_data.normalize_obs(ensemble_samples[..., :-1])
            obs_norm, act_norm, next_obs_norm = map(
                lambda x: torch.from_numpy(x).to(torch.get_default_dtype()).to(self._dynamics_rewards.device),
                [obs_norm, act_norm, next_obs_norm])
            _num_models = next_obs_norm.shape[0]
            obs_norm = obs_norm.unsqueeze(0).repeat_interleave(_num_models, 0)
            act_norm = act_norm.unsqueeze(0).repeat_interleave(_num_models, 0)
            # z is a constant in MuJoCo tasks, it is useful when there are hidden factor that influence dynamics transitions but not observable by policy.
            z_shape = list(obs_norm.shape)
            z_shape[-1] = 1
            z = torch.zeros(z_shape)
            z = z.to(self._dynamics_rewards.device)
            ndim0, ndim1, ndim2 = obs_norm.shape
            obs_norm, z, act_norm, next_obs_norm = map(lambda x: x.reshape((-1, x.shape[-1])), [obs_norm, z, act_norm, next_obs_norm])

            with torch.no_grad():
                dynamics_reward = historical_transition_reward_multi_batch(self._dynamics_rewards, obs_norm, z, act_norm,
                                                               next_obs_norm, batch_num=self._reward_infer_batch_num, clip_value=self._d_clip)

            dynamics_reward = dynamics_reward.reshape((ndim0, -1, 1))

            reward_mean = dynamics_reward.mean(dim=0, keepdim=False)
            info['dynamics_reward_mean'] = reward_mean.mean().item()
            if self._reward_use_type == 'softmax_elite':
                # x 10: (temperature of the softmax be 0.1)
                dynamics_reward_softmax = torch.nn.functional.softmax(dynamics_reward.squeeze(-1) * 10, dim=0, )
                dynamics_reward_softmax_t = dynamics_reward_softmax.t()
                dist = Categorical(dynamics_reward_softmax_t)
                indices = dist.sample()
            else:
                indices = torch.max(dynamics_reward, dim=0).indices
            model_idxs = indices.reshape((-1,)).detach().cpu().numpy()
            info['detail_data'] = dict(
                dynamics_reward=dynamics_reward.detach().cpu().numpy(),
                obs=obs,
                act=action,
                next_obs=ensemble_samples[..., :-1],
                rew=ensemble_samples[..., -1:],
                selected_model_idxs=model_idxs
            )
            samples = ensemble_samples[model_idxs, np.arange(batch_size)]
            sampled_rewards = info['detail_data']['dynamics_reward'][model_idxs, np.arange(batch_size)]
        else:

            if deterministic:
                ensemble_samples = mean.astype(np.float32)
            else:
                ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)
            # avoid diverge during sampling
            ensemble_samples = np.clip(ensemble_samples, -clipping_value, clipping_value)

            # choose one model from ensemble
            num_models, batch_size, _ = ensemble_samples.shape
            model_idxs = self.model.random_elite_idxs(batch_size)
            samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        if self._reward_use_type in ['select_elite', 'softmax_elite'] and self._dynamics_rewards is not None:
            # early termination here
            early_stop = (~terminal) & (sampled_rewards < self._terminal_dynamics_reward)
            if current_step is not None and current_step <= self._minimal_rollout_length:
                early_stop[:] = False
            else:
                terminal = terminal | (sampled_rewards < self._terminal_dynamics_reward)
        info["raw_reward"] = np.mean(reward)
        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = np.mean(penalty)

        return next_obs, reward, terminal, info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        # samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        # accelerating
        samples = mean[None, :] + torch.randn((num_samples,) + mean.shape, device=std.device, dtype=std.dtype) * std[None, :]

        next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets
    @staticmethod
    def get_memory_cost():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_cost = memory_info.rss / (1024 * 1024)
        return memory_cost
    def train(
        self,
        data: Dict,
        slogger: sLogger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        start_time = time.time()
        memory_costs = []
        while True:
            memory_costs.append(self.get_memory_cost())
            epoch += 1
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()

            slogger.add_tabular_data(tb_prefix='loss', dynamics_train_loss=train_loss, dynamics_holdout_loss=holdout_loss, iteration=epoch)
            slogger.dump_tabular()
            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break
        end_time = time.time()
        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(slogger.output_dir)
        self.model.eval()
        slogger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
        slogger.log(f'time period: {end_time-start_time}')
        slogger.log(f'maximum memory cost: {max(memory_costs)} MB')

    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)

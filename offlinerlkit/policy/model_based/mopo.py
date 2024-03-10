import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import SACPolicy
from offlinerlkit.dynamics import BaseDynamics


class MOPOPolicy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        state_clip: float = 25,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )
        self._state_clip = state_clip
        self.dynamics = dynamics

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        info_list = {}
        for step in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions,
                                                                             clipping_value=self._state_clip, current_step=step+1)
            if 'early_stop' in info:
                not_early_stop = (~info['early_stop']).flatten()
                info.pop('early_stop')
                rollout_transitions["terminals"].append(terminals[not_early_stop])

                rollout_transitions["obss"].append(observations[not_early_stop])
                rollout_transitions["next_obss"].append(next_observations[not_early_stop])
                rollout_transitions["actions"].append(actions[not_early_stop])
                rollout_transitions["rewards"].append(rewards[not_early_stop])
            else:
                rollout_transitions["terminals"].append(terminals)
                rollout_transitions["obss"].append(observations)
                rollout_transitions["next_obss"].append(next_observations)
                rollout_transitions["actions"].append(actions)
                rollout_transitions["rewards"].append(rewards)

            for k, v in info.items():
                if k not in info_list:
                    info_list[k] = []
                info_list[k].append(v)
            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)
        info_list['num_transitions'] = num_transitions
        info_list['reward_mean'] = rewards_arr.mean()
        return rollout_transitions, info_list

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        return super().learn(mix_batch)

import sys
import time
import os

import numpy as np
import torch
from gym.logger import set_level
set_level(40)
import gym
import copy
from d4rl import gym_mujoco
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy
from smart_logger import  Logger as sLogger
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from offlinerlkit.utils.plot_util import plot_rollout_figure, plot_correcting_data
import platform
import psutil

# model-based policy trainer
class MBPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_policy: BasePolicy,
        eval_env: gym.Env,
        dynamics_eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        slogger: sLogger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0,
        dynamics_rewards = None,
        process_rnd_generator = None,
    ) -> None:
        self.policy = policy
        self.eval_policy = eval_policy
        self.eval_env = eval_env
        self.dynamics_eval_env = dynamics_eval_env
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.slogger = slogger
        self._rollout_freq, self._rollout_batch_size, \
            self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq
        self._process_rnd_generator = process_rnd_generator

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._eval_episodes = eval_episodes
        self._dynamics_rewards = dynamics_rewards
        self.lr_scheduler = lr_scheduler
        # self._process_pool = ProcessPoolExecutor(max_workers=4)
        # self._thread_pool = ThreadPoolExecutor(max_workers=2)

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):
            self.policy.train()
            original_device = self.policy.actor.device
            self.policy.to('cpu')
            self.eval_policy.actor.load_state_dict(copy.deepcopy(self.policy.actor.state_dict()))
            self.eval_policy.to('cpu')
            # future = self._process_pool.submit(MBPolicyTrainer._evaluate_process, self.eval_policy, self.eval_env, self._eval_episodes, self._process_rnd_generator.randint(0, 100000000))
            eval_info = self._evaluate_func(self.eval_policy, self.eval_env, self._eval_episodes)
            self.policy.to(original_device)

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                if num_timesteps % self._rollout_freq == 0:
                    init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
                    self.fake_buffer.add_batch(**rollout_transitions)
                    self.slogger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    )


                    for _key, _value in rollout_info.items():
                        if not isinstance(_value, list):
                            self.slogger.add_tabular_data(tb_prefix='rollout_info', **{_key: _value})
                        else:
                            if len(_value) > 0 and isinstance(_value[0], dict):
                                pass
                            else:
                                self.slogger.add_tabular_data(tb_prefix='rollout_info', **{_key: _value})
                self.slogger.add_tabular_data(tb_prefix='rollout_info', average_rollout_length=rollout_info['num_transitions']/self._rollout_batch_size)
                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size
                real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                self.slogger.add_tabular_data(tb_prefix='rollout_info', **loss)

                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                    dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
                    self.slogger.add_tabular_data(**dynamics_update_info)
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            # eval_info = future.result()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            if hasattr(self.eval_env, 'get_normalized_score'):
                norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            else:
                env_name = self.eval_env.get_name()

                data_name_tmp = env_name.split('-')[0].lower()
                name_to_min_max = {
                    'walker2d': (1, 5143),
                    'hopper': (5, 3294),
                    'halfcheetah': (-298, 12284),
                }
                # d4rl
                # name_to_min_max = {
                #     'walker2d': (1.62, 4592.3),
                #     'hopper': (20.272305, 3234.3),
                #     'halfcheetah': (-280.178953, 12135.0 ),
                # }
                min_ref = name_to_min_max[data_name_tmp][0]
                max_ref = name_to_min_max[data_name_tmp][1]
                norm_ep_rew_mean = (ep_reward_mean - min_ref) / (max_ref - min_ref) * 100
                norm_ep_rew_std = ep_reward_std / (max_ref - min_ref) * 100
            last_10_performance.append(norm_ep_rew_mean)
            self.slogger.add_tabular_data(tb_prefix='eval', norm_ep_rew_mean=norm_ep_rew_mean,
                                          norm_ep_rew_std=norm_ep_rew_std, ep_length_mean=ep_length_mean,
                                          ep_length_std=ep_length_std, iteration=e, timestep=num_timesteps)
            self.slogger.dump_tabular()
            # save checkpoint
            self.slogger(f'output dir: {self.slogger.output_dir}')
            if e % 50 == 0 and platform.system() == 'Darwin':
                self.slogger.sync_log_to_remote(trial_num=5)

        self.slogger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.slogger.output_dir, "policy.pth"))
        self.policy.dynamics.save(self.slogger.output_dir)

        return {"last_10_performance": np.mean(last_10_performance)}

    @staticmethod
    def _evaluate_process(policy, eval_env: gym.Env, eval_episodes, seed=None) -> Dict[str, List[float]]:
        import random
        import numpy as np
        import torch
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            eval_env.seed(seed)
            eval_env.observation_space.seed(seed)
            eval_env.action_space.seed(seed)
        policy.eval()
        obs = eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        while num_episodes < eval_episodes:
            action = policy.select_action(obs.reshape(1, -1), deterministic=True)
            next_obs, reward, terminal, _ = eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes += 1
                episode_reward, episode_length = 0, 0
                obs = eval_env.reset()

        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    @staticmethod
    def _evaluate_func(policy, eval_env: gym.Env, eval_episodes) -> Dict[str, List[float]]:
        with torch.no_grad():
            policy.eval()
            obs = eval_env.reset()
            eval_ep_info_buffer = []
            num_episodes = 0
            episode_reward, episode_length = 0, 0
            while num_episodes < eval_episodes:
                action = policy.select_action(obs.reshape(1, -1), deterministic=True)
                next_obs, reward, terminal, _ = eval_env.step(action.flatten())
                episode_reward += reward
                episode_length += 1

                obs = next_obs

                if terminal:
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward, "episode_length": episode_length}
                    )
                    num_episodes += 1
                    episode_reward, episode_length = 0, 0
                    obs = eval_env.reset()

            return {
                "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
                "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
            }

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1, -1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
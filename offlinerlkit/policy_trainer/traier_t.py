import os

import neorl
import numpy as np

from offlinerlkit.utils.plot_util import plot_rollout_figure
import smart_logger
from offlinerlkit.common_config.load_config import init_smart_logger
from models.discriminator_ensemble_loader import load_discriminator_ensemble, historical_transition_reward, historical_transition_reward_multi_batch, DiscriminatorEnsemble, historical_transition_reward_iteration
import torch
import os
import gym
from d4rl import gym_mujoco


def get_real_next_obs(obs, action, env_name):
    if env_name.endswith('-v3-H') or env_name.endswith('-v3-M') or env_name.endswith('-v3-L'):
        env = neorl.make(env_name[:-2])
    else:
        env = gym.make(env_name)
    env.reset()
    obs_num = obs.shape[0]
    next_obss = []
    for i in range(obs_num):
        qpos = env.unwrapped.data.qpos.flat[:]
        qvel = env.unwrapped.data.qvel.flat[:]
        if len(qpos) + len(qvel) == obs.shape[-1]:
            qpos[:] = obs[i, :qpos.shape[0]]
            qvel[:] = obs[i, qpos.shape[0]:]
        else:
            qpos[1:] = obs[i, :qpos.shape[0] - 1]
            qvel[:] = obs[i, qpos.shape[0] - 1:]
        env.unwrapped.set_state(qpos, qvel)
        real_next_obs, reward, done, _ = env.step(action[i])
        real_next_obs = np.array(real_next_obs).reshape((1, -1))
        next_obss.append(real_next_obs)
        if done:
            env.reset()
    next_obss = np.concatenate(next_obss, axis=0)
    return next_obss


def main():
    init_smart_logger()
    env_name = 'walker2d-medium-replay-v2'
    env_name = 'Walker2d-v3-H'
    # env_name = 'hopper-medium-replay-v2'
    discriminator_ensemble = load_discriminator_ensemble(env_name)
    discriminator_ensemble_a = DiscriminatorEnsemble(discriminator_ensemble, 10, 400, load_min_max=False)
    discriminator_ensemble_a.final_reward_test()
    # discriminator_ensemble_a.max_expert_value = 1.0
    # discriminator_ensemble_a.minimum_learner_value = 0.0
    # exit(0)
    test_data_path = os.path.join(smart_logger.get_base_path(), 'dynamics_reward_models', env_name, 'test_detail_data.npz')
    data = np.load(test_data_path)
    keys = [k for k, v in data.items()]
    output_dir = os.path.join(smart_logger.get_base_path(), 'tmp_folder')
    for k in keys:
        print(f'k: {k}, shape: {data[k].shape}')
    detail_data_dynamics_reward = data['detail_data_dynamics_reward']
    detail_data_obs = data['detail_data_obs']
    detail_data_act = data['detail_data_act']
    detail_data_next_obs = data['detail_data_next_obs']
    detail_data_rew = data['detail_data_rew']
    selected_model_idxs = data['selected_model_idxs']
    detail_data_dynamics_mae = data['detail_data_dynamics_mae']
    detail_data_reward_mae = data['detail_data_reward_mae']
    detail_real_next_obs = get_real_next_obs(detail_data_obs, detail_data_act, env_name)
    obs, act, next_obs, real_next_obs = map(lambda x: torch.from_numpy(x).to(torch.get_default_dtype()), [detail_data_obs,
                                                                                           detail_data_act,
                                                                                           detail_data_next_obs,
                                                                                           detail_real_next_obs])
    obs = discriminator_ensemble.expert_data.normalize_obs(obs)

    next_obs = discriminator_ensemble.expert_data.normalize_obs(next_obs)
    real_next_obs = discriminator_ensemble.expert_data.normalize_obs(real_next_obs)
    act = discriminator_ensemble.expert_data.normalize_act(act)

    _num_models = next_obs.shape[0]
    obs = obs.unsqueeze(0).repeat_interleave(_num_models, 0)
    act = act.unsqueeze(0).repeat_interleave(_num_models, 0)
    z_shape = list(obs.shape)
    z_shape[-1] = 1
    z = torch.zeros(z_shape)
    z = z.to(obs.device)
    print(obs.shape, z.shape, act.shape, next_obs.shape, real_next_obs.shape)
    real_data_reward = historical_transition_reward_multi_batch(discriminator_ensemble_a, obs[0], z[0], act[0], real_next_obs, 200)
    # real_data_reward = historical_transition_reward_iteration(discriminator_ensemble, obs[0], z[0], act[0], real_next_obs)
    discriminator_ensemble_a.historical_reward_test(obs[0], act[0], z[0], real_next_obs)
    real_data_reward = real_data_reward.unsqueeze(0).repeat_interleave(_num_models, 0)
    obs, act, z, next_obs = map(lambda x: x.reshape((-1, x.shape[-1])), [obs, act, z, next_obs])
    reward = historical_transition_reward_multi_batch(discriminator_ensemble_a, obs, z, act, next_obs, 100)
    # reward = historical_transition_reward_iteration(discriminator_ensemble, obs, z, act, next_obs, ensemble_strategy='prob_mean_clip', clip_reward=True)

    reward = reward.detach().cpu().numpy()
    print('reward shape', reward.shape)
    plot_rollout_figure(output_dir, detail_data_dynamics_reward.reshape((-1,)), detail_data_dynamics_mae.reshape((-1,)), 'full_state_mae.png')
    plot_rollout_figure(output_dir, reward.reshape((-1,)), detail_data_dynamics_mae.reshape((-1,)), 'rerun_state_mae.png')
    plot_rollout_figure(output_dir, real_data_reward.reshape((-1,)), detail_data_dynamics_mae.reshape((-1,)), 'rerun_real_state_mae.png')
    plot_rollout_figure(output_dir, detail_data_dynamics_reward.reshape((-1,)), detail_data_reward_mae.reshape((-1,)), 'full_reward_mae.png')
    pass



if __name__ == '__main__':
    main()
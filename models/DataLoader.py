import os
import numpy as np
import torch
import gym
import d4rl.gym_mujoco
import time
import neorl

def convert_d4rl(data_name):
    env = gym.make(data_name[5:])
    for i in range(80):
        try:
            data = env.get_dataset()
            break
        except PermissionError as e:
            print(f'load fail, reloading')
            time.sleep(5)
        except BlockingIOError as e:
            print(f'load fail, reloading')
            time.sleep(5)
    else:
        raise PermissionError(f'load fail!!!')
    # for k, v in data.items():
    #     print(k, v.shape)
    data_num = data["observations"].shape[0]

    ## get min_len, max_len
    min_len = int(1e10)
    max_len = 0
    i = j = 0
    total_length = 0
    num_traj = 0
    while True:
        if j==data_num-1:
            tmp_len = j - i + 1
            ###################
            total_length += tmp_len
            num_traj += 1
            ###################
            min_len = min(tmp_len, min_len)
            max_len = max(tmp_len, max_len)
            break
        elif data["terminals"][j] or data["timeouts"][j]:
            tmp_len = j - i + 1
            ###################
            total_length += tmp_len
            num_traj += 1
            ###################
            min_len = min(tmp_len, min_len)
            max_len = max(tmp_len, max_len)
            j += 1
            i = j
        else:
            j += 1
    assert total_length==data_num
    # print("min_len:{}".format(min_len))
    print("shape: ({}, {}, ...)".format(num_traj, max_len))

    data_dict = {}
    data_dict["observations"] = np.zeros((num_traj, max_len, data["observations"].shape[-1]))
    data_dict["next_observations"] = np.zeros((num_traj, max_len, data["next_observations"].shape[-1]))
    data_dict["actions"] = np.zeros((num_traj, max_len, data["actions"].shape[-1]))
    data_dict["next_actions"] = np.zeros((num_traj, max_len, data["actions"].shape[-1]))
    data_dict["rewards"] = np.zeros((num_traj, max_len))
    data_dict["dones"] = np.zeros((num_traj, max_len))
    data_dict["masks"] = np.zeros((num_traj, max_len))
    data_dict["traj_len"] = np.zeros((num_traj, ))

    def fill_data(data_dict, data, line_index, fill_length, i, j):
        data_dict["observations"][line_index, :fill_length, :] = data["observations"][i:j+1, :].copy()
        data_dict["next_observations"][line_index, :fill_length, :] = data["next_observations"][i:j+1, :].copy()
        data_dict["actions"][line_index, :fill_length, :] = data["actions"][i:j+1, :].copy()
        data_dict["next_actions"][line_index, :fill_length-1, :] = data["actions"][i+1:j+1, :].copy()
        data_dict["next_actions"][line_index, fill_length-1, :] = data["actions"][j, :].copy()
        data_dict["rewards"][line_index, :fill_length] = data["rewards"][i:j+1].copy()
        data_dict["dones"][line_index, fill_length-1] = True
        data_dict["masks"][line_index, :fill_length] = np.ones((fill_length))
        data_dict["traj_len"][line_index] = fill_length


    ## convert dataset
    i = j = 0
    total_length = 0
    num_traj = 0
    while True:
        if j==data_num-1:
            tmp_len = j - i + 1
            fill_data(data_dict, data, num_traj, tmp_len, i, j)
            ###################
            total_length += tmp_len
            num_traj += 1
            ###################
            break
        elif data["terminals"][j] or data["timeouts"][j]:
            tmp_len = j - i + 1
            fill_data(data_dict, data, num_traj, tmp_len, i, j)
            ###################
            total_length += tmp_len
            num_traj += 1
            ###################
            j += 1
            i = j
        else:
            j += 1
    assert total_length==data_num
    # os.makedirs('./d4rl_dataset_converted', exist_ok=True)
    # np.save("./d4rl_dataset_converted/{}".format(data_name), data_dict)
    # print('"{}" converted successfully!'.format(data_name))
    return data_dict
def convert_neorl(data_name):
    data_class, env_name, env_version, data_type = data_name.split('-')
    env = neorl.make(f'{env_name}-{env_version}')
    data, _ = env.get_dataset(data_type=data_type, train_num=1000, need_val=False)
    data_lens = []
    lst_idx = 0
    for i in data['index'][1:]:
        data_lens.append(i - lst_idx)
        lst_idx = i
    data_lens.append(data['obs'].shape[0] - lst_idx)
    max_len = np.max(data_lens)
    num_traj = len(data['index'])

    data_dict = {}
    data_dict["observations"] = np.zeros((num_traj, max_len, data["obs"].shape[-1]))
    data_dict["next_observations"] = np.zeros((num_traj, max_len, data["next_obs"].shape[-1]))
    data_dict["actions"] = np.zeros((num_traj, max_len, data["action"].shape[-1]))
    data_dict["next_actions"] = np.zeros((num_traj, max_len, data["action"].shape[-1]))
    data_dict["rewards"] = np.zeros((num_traj, max_len))
    data_dict["dones"] = np.zeros((num_traj, max_len))
    data_dict["masks"] = np.zeros((num_traj, max_len))
    data_dict["traj_len"] = np.zeros((num_traj,))

    for line_index, (idx, fill_length) in enumerate(zip(data['index'], data_lens)):
        data_dict["observations"][line_index, :fill_length, :] = data["obs"][idx:idx+fill_length, :].copy()
        data_dict["next_observations"][line_index, :fill_length, :] = data["next_obs"][idx:idx+fill_length, :].copy()
        # TODO: clip action to [-1, 1] here
        data_dict["actions"][line_index, :fill_length, :] = np.clip(data["action"][idx:idx+fill_length, :].copy(), -1, 1)
        data_dict["next_actions"][line_index, :fill_length - 1, :] = np.clip(data["action"][idx+1:idx+fill_length, :].copy(), -1, 1)
        data_dict["next_actions"][line_index, fill_length - 1, :] = np.clip(data["action"][idx+fill_length-1, :].copy(), -1, 1)
        data_dict["rewards"][line_index, :fill_length] = data["reward"][idx:idx+fill_length, 0].copy()
        data_dict["dones"][line_index, fill_length - 1] = True
        data_dict["masks"][line_index, :fill_length] = np.ones((fill_length,))
        data_dict["traj_len"][line_index] = fill_length
    return data_dict

class DataLoader:
    def __init__(self, data):
        # data_path = "./d4rl_dataset_converted/{}.npy".format(data)
        # if not os.path.exists(data_path):
        #     convert_d4rl(data)
        # else:
        #     print('"{}" has been saved in "./d4rl_dataset_converted"!'.format(data))
        if data.endswith('-v3-L') or data.endswith('-v3-M') or data.endswith('-v3-H'):
            self.data = convert_neorl(data)
        else:
            self.data = convert_d4rl(data)
        # self.data = np.load(data_path, allow_pickle=True).item()
        for k, v in self.data.items():
            print(k, v.shape)

        self.obs = self.data['observations']
        self.next_obs = self.data['next_observations']
        self.act = self.data['actions']
        self.next_act = self.data['next_actions']
        self.reward = self.data['rewards']
        self.done = self.data['dones']
        self.mask = self.data['masks']
        self.traj_len = self.data['traj_len']
        ####
        self.min_post_normal_value = -0.8
        self.max_post_normal_value = 0.8
        self.min_post_normal_value_act = -1.0
        self.max_post_normal_value_act = 1.0
        self.z = np.zeros((self.obs.shape[0], self.obs.shape[1], 1))
        self.next_z = np.zeros((self.obs.shape[0], self.obs.shape[1], 1))
        print(f'obs shape: {self.obs.shape}, next obs shape: {self.next_obs.shape}, act shape: {self.act.shape}')
        self.obs_dim = self.obs.shape[-1]
        self.act_dim = self.act.shape[-1]
        self.hidden_dim = self.z.shape[-1]
        self.obs, self.obs_min, self.obs_max = self.min_max_normalization(self.obs, mask=self.mask)
        self.next_obs, _, _ = self.min_max_normalization(self.next_obs, min_v=self.obs_min, max_v=self.obs_max)
        self.act, self.act_min, self.act_max = self.min_max_normalization(self.act, mask=self.mask, min_target=self.min_post_normal_value_act, max_target=self.max_post_normal_value_act)
        self.obs_ext = np.concatenate((self.obs, self.z), axis=-1)
        self.split_obs = []
        self.split_act = []
        self.split_next_obs = []
        print(f'obs max: {self.obs_max}, obs min: {self.obs_min}, act max: {self.act_max}, act min: {self.act_min}')
        self.obs_cnvt = self.obs.reshape((-1, self.obs.shape[-1]))
        self.act_cnvt = self.act.reshape((-1, self.act.shape[-1]))
        self.next_act_cnvt = self.next_act.reshape((-1, self.act.shape[-1]))
        self.next_obs_cnvt = self.next_obs.reshape((-1, self.next_obs.shape[-1]))
        self.reward_cnvt = self.reward.reshape((-1, 1))
        self.done_cnvt = self.done.reshape((-1, 1))
        self.mask_cnvt = self.mask.reshape((-1, 1))
        self.z_cnvt = self.z.reshape((-1, self.z.shape[-1]))
        self.next_z_cnvt = self.next_z.reshape((-1, self.next_z.shape[-1]))
        self.obs_cnvt, self.z_cnvt, self.act_cnvt, self.next_obs_cnvt, self.next_z_cnvt, self.next_act_cnvt, self.traj_len, self.mask, self.mask_cnvt = map(
            lambda x: torch.Tensor(x).to(torch.device('cpu')),
            [self.obs_cnvt, self.z_cnvt, self.act_cnvt, self.next_obs_cnvt, self.next_z_cnvt, self.next_act_cnvt, self.traj_len, self.mask,
             self.mask_cnvt])
        self.obs_max_tensor = torch.from_numpy(self.obs_max).to(torch.float)
        self.obs_min_tensor = torch.from_numpy(self.obs_min).to(torch.float)
        self.act_max_tensor = torch.from_numpy(self.act_max).to(torch.float)
        self.act_min_tensor = torch.from_numpy(self.act_min).to(torch.float)
        # self.split_data()
        # self.choose_data_idx(2)

    def to_device(self, device):
        self.obs_cnvt, self.z_cnvt, self.act_cnvt, self.next_obs_cnvt, self.next_z_cnvt, self.next_act_cnvt, self.traj_len, self.mask, self.mask_cnvt, \
            self.mask_cnvt = map(
            lambda x: x.to(device),
            [self.obs_cnvt, self.z_cnvt, self.act_cnvt, self.next_obs_cnvt, self.next_z_cnvt, self.next_act_cnvt,
             self.traj_len, self.mask, self.mask_cnvt,
             self.mask_cnvt])
        self.obs_max_tensor, self.obs_min_tensor, self.act_max_tensor, self.act_min_tensor = map(lambda x: x.to(device), [
            self.obs_max_tensor, self.obs_min_tensor, self.act_max_tensor, self.act_min_tensor
        ])

    def min_max_normalization(self, x:np.ndarray, mask=None, min_v=None, max_v=None, min_target=None, max_target=None):
        if min_v is None:
            assert mask is not None
            assert len(x.shape)==3
            mask_flat = mask.reshape((-1, ))
            x_flat = x.reshape((-1, x.shape[-1]))
            valid_index = np.nonzero(mask_flat)
            x_flat_valid = x_flat[valid_index[0]]
            min_v = np.min(x_flat_valid, axis=0)
        if max_v is None:
            assert mask is not None
            assert len(x.shape)==3
            mask_flat = mask.reshape((-1, ))
            x_flat = x.reshape((-1, x.shape[-1]))
            valid_index = np.nonzero(mask_flat)
            x_flat_valid = x_flat[valid_index[0]]
            max_v = np.max(x_flat_valid, axis=0)
        min_target = self.min_post_normal_value if min_target is None else min_target
        max_target = self.max_post_normal_value if max_target is None else max_target
        x = (x - min_v) / (max_v - min_v) * (max_target - min_target) + min_target
        return x, min_v, max_v

    def unormalize_act(self, act):
        act_max = self.act_max
        act_min = self.act_min
        if isinstance(act, torch.Tensor):
            act_max = self.act_max_tensor
            act_min = self.act_min_tensor
        act = (act - self.min_post_normal_value_act) / (self.max_post_normal_value_act - self.min_post_normal_value_act) * (act_max- act_min) + act_min
        return act

    def normalize_act(self, act):
        act_max = self.act_max
        act_min = self.act_min
        if isinstance(act, torch.Tensor):
            act_max = self.act_max_tensor
            act_min = self.act_min_tensor
        act = (act - act_min) / (act_max - act_min) * (self.max_post_normal_value_act - self.min_post_normal_value_act) + self.min_post_normal_value_act
        return act

    def unormalize_obs(self, obs):
        obs_max = self.obs_max
        obs_min = self.obs_min
        if isinstance(obs, torch.Tensor):
            obs_max = self.obs_max_tensor
            obs_min = self.obs_min_tensor
        obs = (obs - self.min_post_normal_value) / (self.max_post_normal_value - self.min_post_normal_value) * (obs_max - obs_min) + obs_min
        return obs

    def normalize_obs(self, obs):
        obs_max = self.obs_max
        obs_min = self.obs_min
        if isinstance(obs, torch.Tensor):
            obs_max = self.obs_max_tensor
            obs_min = self.obs_min_tensor
        return self.min_max_normalization(obs, min_v=obs_min, max_v=obs_max)[0]

    def print_data(self):
        print(self.z.tolist())
        print(np.hstack((self.obs, self.next_obs)))

    def split_data(self):
        z_diff = np.diff(self.z.reshape((-1,)))
        end_point = np.where(z_diff == 1)[0] + 1
        start_point = np.where(z_diff == -1)[0] + 1
        print(f'len of end points: {len(end_point)}, len of start points {len(start_point)}')
        start_idx = 0
        for i in range(len(end_point)):
            self.split_obs.append(self.obs[start_idx:end_point[i]])
            self.split_act.append(self.act[start_idx:end_point[i]])
            self.split_next_obs.append(self.next_obs[start_idx:end_point[i]])
            start_idx = start_point[i]

    def choose_data_idx(self, idx):
        self.obs = self.split_obs[idx]
        self.act = self.split_act[idx]
        self.next_obs = self.split_next_obs[idx]

    def scatter_data(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.obs, self.act)
        plt.show()

    def norm_raw_obs(self, obs):
        return obs

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.obs_cnvt.shape[0], (batch_size,))
        state = self.obs_cnvt[idxs]
        action = self.act_cnvt[idxs]
        next_state = self.next_obs_cnvt[idxs]
        z = self.z_cnvt[idxs]
        next_act = self.next_act_cnvt[idxs]
        return state, z, action, next_state, next_act

    def sample_with_reward(self, batch_size):
        idxs = np.random.randint(0, self.obs_cnvt.shape[0], (batch_size,))
        state = self.obs_cnvt[idxs]
        action = self.act_cnvt[idxs]
        next_state = self.next_obs_cnvt[idxs]
        z = self.z_cnvt[idxs]
        next_act = self.next_act_cnvt[idxs]
        reward = self.reward_cnvt[idxs]
        return state, z, action, next_state, next_act, reward

    def to(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    # loader = DataLoader('cool_control_2000_0211.npz')
    loader = DataLoader('d4rl-hopper-medium-v2')
    # print(f'obs dim: {loader.obs_dim}, act dim: {loader.act_dim}, z dim: {loader.hidden_dim}')
    # print(loader.next_obs - loader.obs)
    # loader = DataLoader('cool_control_2000.npzconverted.npz')
    # loader = DataLoader('/Users/fanmingluo/PycharmProjects/refrigeration_reconstruction/cool_control_2000.npzconverted.npy')
    # loader.convert_data()
    # loader.print_data()
    # loader.plot_data()
    # loader.split_data()
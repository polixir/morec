import matplotlib
matplotlib.use('Agg')
# use Agg to avoid memory leak!!!
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def plot_rollout_figure(output_dir, transition_reward, real_mae,
                        file_name='rollout_reward_curve.png', point_alpha=1.0, folder_name='pic'):
        figsize = (3.2 * 2 * 5, 2.24 * 2)
        f, axarr = plt.subplots(1, 5, sharex=False, sharey=False, squeeze=False,
                                figsize=figsize)
        # ----
        ax = axarr[0][0]
        ax.plot(transition_reward, 'r', label='transition reward')
        ax.set_ylabel('transition_reward', color='r')
        ax.tick_params(colors='r')
        ax.set_title(f'model rollout data reward curve, MAE: {round(real_mae.mean(), 4)}')

        ax2 = ax.twinx()
        ax2.plot(real_mae, 'b', label='mae')
        ax2.set_ylabel('mae', color='b')
        ax2.tick_params(colors='b')

        real_mae_sorted_idx = np.argsort(real_mae)
        ax.legend()
        ax = axarr[0][1]

        ax.scatter(real_mae, transition_reward, alpha=point_alpha, edgecolors='none')

        # ax.plot(real_mae[real_mae_sorted_idx], transition_reward[real_mae_sorted_idx])
        ax.set_xlabel('mae')
        ax.set_ylabel('transition_reward')
        corr_matrix = np.corrcoef(np.log(real_mae + 1e-9), transition_reward)
        # print(f'CORR: {corr_matrix}')
        ax.set_title(f'transition_reward----MAE, {corr_matrix[0, 1]}')
        ax.set_xscale('log')

        ax = axarr[0][2]
        # ax.plot(real_mae[real_mae_sorted_idx], real_rollout_transition_reward[real_mae_sorted_idx])
        ax.set_xlabel('mae')
        ax.set_ylabel('real_transition_reward')
        ax.set_title('real_transition_reward----MAE')
        ax.set_xscale('log')

        ax = axarr[0][3]
        try:
            g = sns.histplot(x=real_mae, y=transition_reward, cbar=True, kde=True, # cmap='Blues',
                             ax=ax, log_scale=(True, False))  # hist参数表示是否显示直方图，默认为True
        except Exception as e:
            pass

        ax.set_xlabel('mae')
        ax.set_ylabel('transition_reward')
        ax.set_title('distribution figure')
        ax.set_xscale('log')

        os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
        f.savefig(os.path.join(output_dir, folder_name, file_name), dpi=300, bbox_inches='tight')
        f.clf()


        plt.close(f)
        # # real_mae = np.log(real_mae + 1e-9)
        # g = sns.jointplot(x=real_mae, y=transition_reward)
        # # g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
        # # g.ax_joint.set_xscale('log')
        #
        # g.plot_joint(sns.kdeplot, color="r", levels=6, log_scale=(True, False), fill=True, alpha=0.6)
        # g.plot_joint(sns.kdeplot, color="r", levels=6, log_scale=(True, False))
        # # sns.set_style("darkgrid")
        # g.ax_joint.grid(True)  # 开启网格线
        # # g.ax_marg_x.grid(True)
        # # g.ax_marg_y.grid(True)
        # f= g.fig
        # plt.xlabel('mae')
        # plt.ylabel('transition_reward')
        # g.savefig(os.path.join(output_dir, folder_name, f'joint_{file_name}'), dpi=300, bbox_inches='tight')
        # plt.close(f)
        return corr_matrix[0, 1]

def smooth(y, radius, mode='two_sided', valid_only=False):
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    else:
        raise NotImplementedError(f'{mode} has not been implemented!')
    return out
def plot_correcting_data(correcting_data, output_dir, file_name, folder_name):
        figsize = (3.2 * 2 , 2.24 * 2)
        f, axarr = plt.subplots(1, 1, sharex=False, sharey=False, squeeze=False,
                                figsize=figsize)
        ax = axarr[0][0]
        smoothed = smooth(correcting_data, radius=10)
        ax.scatter(np.arange(correcting_data.shape[0]), correcting_data)
        ax.plot(np.arange(correcting_data.shape[0]), smoothed)
        ax.set_ylabel('correct_flag')
        ax.set_xlabel('index')
        ax.set_title('selecting correct statistics')
        os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
        f.savefig(os.path.join(output_dir, folder_name, file_name), dpi=300, bbox_inches='tight')
        f.clf()

        plt.close(f)
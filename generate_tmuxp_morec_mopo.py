import copy

import yaml
import os
import math
from smart_logger.scripts.generate_tmuxp_base import generate_tmuxp_file, make_cmd_array
import argparse
import subprocess


MAX_SUBWINDOW =2
MAX_PARALLEL = 2


def get_gpu_count():
    try:
        sp = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str = sp.communicate()
        out_list = out_str[0].decode('utf-8').split('\n')
        out_list = [x for x in out_list if x]
        return len(out_list)
    except Exception as _:
        return 0



def get_cmd_array(total_machine=8, machine_idx=0):
    """
    :return: cmd array: list[list[]], the item in the i-th row, j-th column of the return value denotes the
                        cmd in the i-th window and j-th sub-window
    """
    session_name = 'OfflineRL'
    # 0. 代码运行路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 1. GPU设置
    GPUS = [_ for _ in range(get_gpu_count())]
    # 2. 环境变量设置
    environment_dict = dict(
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=current_path,
        OMP_NUM_THREADS=1,
        D4RL_DATASET_DIR=os.path.join(current_path, 'pretrained', 'd4rl_dataset', 'datasets'),
        D4RL_SUPPRESS_IMPORT_ERROR=1
    )
    directory = current_path
    # 3. 启动脚本
    start_up_header = "python run_example/run_morec_mopo.py "
    # 4. 基础参数
    parameters_base = dict(
        algo_name='mopo',
        ensemble_choosing_interval=10,
        max_retain_num=400,
        reward_infer_batch_num=4,
        rollout_batch_size=2000,
        dynamics_reward_path=os.path.join(current_path, 'pretrained', 'dynamics_reward_models'), # 0713
        load_min_max=False,
        auto_alpha=True,
        d_clip=1.0,
        minimal_rollout_length=0,
        penalty_coef=0.1,
        resample_num_in_transition=2,
        reward_use_type='softmax_elite',
        use_dynamics_reward=True,
        terminal_dynamics_reward=0.6,
        rollout_length=100,
    )
    # 5. 遍历设置
    exclusive_candidates = dict(
        # seed=[1, 2, 3, 4, 5],
        seed=[1],
        task=[
            'hopper-random-v2',
            'hopper-medium-v2',
            'hopper-medium-expert-v2',
            'hopper-medium-replay-v2',

            'walker2d-random-v2',
            'walker2d-medium-v2',
            'walker2d-medium-expert-v2',

            'walker2d-medium-replay-v2',

            'halfcheetah-medium-v2',
            'halfcheetah-medium-replay-v2',
            'halfcheetah-random-v2',
            'halfcheetah-medium-expert-v2',

            'HalfCheetah-v3-L',
            'HalfCheetah-v3-M',
            'HalfCheetah-v3-H',

            'Hopper-v3-L',
            'Hopper-v3-M',
            'Hopper-v3-H',

            'Walker2d-v3-L',
            'Walker2d-v3-M',
            'Walker2d-v3-H',
        ],

    )
    # 6. 单独设置
    aligned_candidates = dict(
        information=['MOREC_RERUN_MOPO_1110'],
        reward_use_type=['select_elite']
    )

    good_config = {
        'halfcheetah-random-v2': {'penalty_coef': 0.5, 'reward_use_type': 'softmax_elite'},
        'halfcheetah-medium-v2': {'penalty_coef': 0.5, 'reward_use_type': 'select_elite'},
        'halfcheetah-medium-replay-v2': {'penalty_coef': 0.5,'reward_use_type': 'softmax_elite'},
        'halfcheetah-medium-expert-v2': {'penalty_coef': 0.5, 'reward_use_type': 'select_elite'},

        'hopper-medium-v2': {'penalty_coef': 15.0, 'reward_use_type': 'softmax_elite'},
        'hopper-medium-replay-v2': {'penalty_coef': 15.0, 'reward_use_type': 'softmax_elite'},
        'hopper-random-v2': {'penalty_coef': 10.0, 'reward_use_type': 'select_elite'},
        'hopper-medium-expert-v2': {'penalty_coef': 15.0, 'reward_use_type': 'softmax_elite'},

        'walker2d-medium-v2': {'penalty_coef': 0.5, 'reward_use_type': 'softmax_elite'},
        'walker2d-medium-expert-v2': {'penalty_coef': 1.0, 'reward_use_type': 'softmax_elite'},
        'walker2d-medium-replay-v2': {'penalty_coef': 0.25, 'reward_use_type': 'select_elite'},
        'walker2d-random-v2': {'penalty_coef': 1.0, 'reward_use_type': 'softmax_elite'},

        'HalfCheetah-v3-H': {'penalty_coef': 2.0, 'reward_use_type': 'softmax_elite'},
        'HalfCheetah-v3-L': {'penalty_coef':2.0, 'reward_use_type': 'select_elite'},
        'HalfCheetah-v3-M': {'penalty_coef':2.0, 'reward_use_type': 'select_elite'},

        'Hopper-v3-H': {'reward_use_type': 'select_elite', 'penalty_coef': 20.0, 'real_ratio': 0.5, 'auto_alpha': False},
        'Hopper-v3-L': {'alpha':0.1, 'reward_use_type': 'select_elite', 'auto_alpha': False, 'penalty_coef': 2.0},
        'Hopper-v3-M': {'reward_use_type': 'select_elite', 'penalty_coef': 20.0, 'real_ratio': 0.5},

        'Walker2d-v3-H': {'penalty_coef': 2.0, 'reward_use_type': 'select_elite'},
        'Walker2d-v3-L': {'penalty_coef': 2.0, 'reward_use_type': 'softmax_elite'},
        'Walker2d-v3-M': {'penalty_coef': 2.0, 'reward_use_type': 'softmax_elite'},
    }

    def task_is_valid(_task):
        if _task['task'] in good_config:
            for k, v in good_config[_task['task']].items():
                _task[k] = v
            _task['load_dynamics_path'] = os.path.join(current_path, 'pretrained', 'learned_dynamics',
                                                       f'{_task["task"]}_seed_{_task["seed"]}-BASELINE')
        return True

    # 从这里开始不用再修改了
    cmd_array, session_name = make_cmd_array(
        directory, session_name, start_up_header, parameters_base, environment_dict,
        aligned_candidates, exclusive_candidates, GPUS, MAX_PARALLEL, MAX_SUBWINDOW,
        machine_idx, total_machine, task_is_valid, split_all=True, sleep_before=0.0, sleep_after=0.0, rnd_seed=42, task_time_interval=60,
    )
    # 上面不用修改了

    # customized command
    # 7. 额外命令
    cmd_array.append(['htop', 'watch -n 1 nvidia-smi'])
    # for win_ind, win_cmds_list in enumerate(cmd_array):
    #     for pane_ind, pand_cmd in enumerate(win_cmds_list):
    #         print(f'win: {win_ind}, pane: {pane_ind}: {pand_cmd}')
    return cmd_array, session_name


def main():
    parser = argparse.ArgumentParser(description=f'generate parallel environment')

    parser.add_argument('--machine_idx', '-idx', type=int, default=-1,
                        help="Server port")
    parser.add_argument('--total_machine_num', '-tn', type=int, default=8,
                        help="Server port")
    args = parser.parse_args()
    cmd_array, session_name = get_cmd_array(args.total_machine_num, args.machine_idx)
    generate_tmuxp_file(session_name+'-2', cmd_array, use_json=True, layout='even-horizontal')


if __name__ == '__main__':
    main()

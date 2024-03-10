import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.*")

import argparse
import os
import sys
import random
import time
from gym.logger import set_level
set_level(40)

import gym
import d4rl
import neorl

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset, load_neorl_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import MOPOPolicy, SACPolicy
from offlinerlkit.common_config.load_config import init_smart_logger
from smart_logger.parameter.ParameterTemplate2 import ParameterTemplate
import smart_logger
import multiprocessing
from datetime import datetime
from smart_logger import Logger as sLogger
from models.discriminator_ensemble_loader import load_discriminator_ensemble, historical_transition_reward, DiscriminatorEnsemble
from offlinerlkit.policy import MOBILEPolicy


class Parameter(ParameterTemplate):
    def __init__(self, config_path=None, debug=False):
        super(Parameter, self).__init__(config_path, debug)

    def parser_init(self):
        parser = argparse.ArgumentParser(description=smart_logger.experiment_config.EXPERIMENT_TARGET)
        self.algo_name = "mobile"
        parser.add_argument("--algo_name", type=str, default=self.algo_name)

        self.task = "walker2d-medium-expert-v2"
        parser.add_argument("--task", type=str, default=self.task)

        self.reward_use_type = 'reward_bonus'
        parser.add_argument("--reward_use_type", type=str, default=self.reward_use_type, choices=['reward_bonus', 'select_elite', 'softmax_elite'])

        self.dynamics_reward_path = None
        parser.add_argument("--dynamics_reward_path", type=str, default=self.dynamics_reward_path,)

        self.seed = 1
        parser.add_argument("--seed", type=int, default=self.seed)
        self.resample_num_in_transition = 1
        parser.add_argument("--resample_num_in_transition", type=int, default=self.resample_num_in_transition)

        self.actor_lr = 1e-4
        parser.add_argument("--actor_lr", type=float, default=self.actor_lr)

        self.critic_lr = 3e-4
        parser.add_argument("--critic_lr", type=float, default=self.critic_lr)

        self.hidden_dims = [256, 256]
        parser.add_argument("--hidden_dims", type=int, nargs='*', default=self.hidden_dims)

        self.gamma = 0.99
        parser.add_argument("--gamma", type=float, default=self.gamma)


        self.tau = 0.005
        parser.add_argument("--tau", type=float, default=self.tau)

        self.alpha = 0.2
        parser.add_argument("--alpha", type=float, default=self.alpha)

        self.auto_alpha = False
        parser.add_argument("--auto_alpha", default=self.auto_alpha, action='store_true')
        self.use_dynamics_reward = False
        parser.add_argument("--use_dynamics_reward", default=self.use_dynamics_reward, action='store_true')
        self.load_min_max = False
        parser.add_argument("--load_min_max", default=self.load_min_max, action='store_true')

        self.target_entropy = None
        parser.add_argument("--target_entropy", type=int, default=self.target_entropy)
        self.alpha_lr = 1e-4
        parser.add_argument("--alpha_lr", type=float, default=self.alpha_lr)
        self.num_q_ensemble = 2
        parser.add_argument("--num_q_ensemble", type=int, default=self.num_q_ensemble)
        self.dynamics_lr = 1e-3
        parser.add_argument("--dynamics_lr", type=float, default=self.dynamics_lr)
        self.num_samples = 10
        parser.add_argument("--num_samples", type=int, default=self.num_samples)
        self.dynamics_hidden_dims = [200, 200, 200, 200]
        parser.add_argument("--dynamics_hidden_dims", type=int, nargs='*', default=self.dynamics_hidden_dims)

        self.dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]
        parser.add_argument("--dynamics_weight_decay", type=float, nargs='*',
                            default=self.dynamics_weight_decay)

        self.n_ensemble = 7
        parser.add_argument("--n_ensemble", type=int, default=self.n_ensemble)

        self.n_elites = 5
        parser.add_argument("--n_elites", type=int, default=self.n_elites)

        self.rollout_freq = 1000
        parser.add_argument("--rollout_freq", type=int, default=self.rollout_freq)

        self.dynamics_max_epochs = None
        parser.add_argument("--dynamics_max_epochs", type=int, default=self.dynamics_max_epochs)

        self.rollout_batch_size = 50000
        parser.add_argument("--rollout_batch_size", type=int, default=self.rollout_batch_size)
        self.rollout_length = 1
        parser.add_argument("--rollout_length", type=int, default=self.rollout_length)
        self.penalty_coef = 2.5
        parser.add_argument("--penalty_coef", type=float, default=self.penalty_coef)
        self.dynamics_reward_factor = 0.99
        parser.add_argument("--dynamics_reward_factor", type=float, default=self.dynamics_reward_factor)
        self.d_clip = 0.999
        parser.add_argument("--d_clip", type=float, default=self.d_clip)
        self.lr_scheduler = True
        parser.add_argument("--lr_scheduler", default=self.lr_scheduler, action='store_true')
        self.deterministic_backup = False
        parser.add_argument("--deterministic_backup", default=self.deterministic_backup, action='store_true')

        self.model_retain_epochs = 5
        parser.add_argument("--model_retain_epochs", type=int, default=self.model_retain_epochs)
        self.ensemble_choosing_interval = 10
        parser.add_argument("--ensemble_choosing_interval", type=int, default=self.ensemble_choosing_interval)
        self.max_retain_num = 200
        parser.add_argument("--max_retain_num", type=int, default=self.max_retain_num)
        self.reward_infer_batch_num = 20
        parser.add_argument("--reward_infer_batch_num", type=int, default=self.reward_infer_batch_num)
        self.real_ratio = 0.05
        parser.add_argument("--real_ratio", type=float, default=self.real_ratio)

        self.terminal_dynamics_reward = 0.6
        parser.add_argument("--terminal_dynamics_reward", type=float, default=self.terminal_dynamics_reward)
        self.load_dynamics_path = None
        parser.add_argument("--load_dynamics_path", type=str, default=self.load_dynamics_path)
        self.epoch = 3000
        parser.add_argument("--epoch", type=int, default=self.epoch)
        self.step_per_epoch = 1000
        parser.add_argument("--step_per_epoch", type=int, default=self.step_per_epoch)
        self.eval_episodes = 10
        parser.add_argument("--eval_episodes", type=int, default=self.eval_episodes)

        self.minimal_rollout_length = 0
        parser.add_argument("--minimal_rollout_length", type=int, default=self.minimal_rollout_length)
        self.batch_size = 256
        parser.add_argument("--batch_size", type=int, default=self.batch_size)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        parser.add_argument("--device", type=str, default=self.device)
        self.dynamics_reward_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        parser.add_argument("--dynamics_reward_device", type=str, default=self.dynamics_reward_device)

        self.information = "INIT"
        parser.add_argument("--information", type=str, default=self.information)
        return parser

def is_neorl(task):
    if task.endswith('-v3-L') or task.endswith('-v3-M') or task.endswith('-v3-H'):
        return True


def train(args, parameters:Parameter):
    # create env and dataset
    if not multiprocessing.get_start_method(allow_none=True) == 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    isneorl = is_neorl(args.task)
    if isneorl:
        env = neorl.make('-'.join(args.task.split('-')[:-1]))
        env_dynamics = neorl.make('-'.join(args.task.split('-')[:-1]))
    else:
        env = gym.make(args.task)
        env_dynamics = gym.make(args.task)
    for i in range(200):
        try:
            if isneorl:
                datatype = args.task.split('-')[-1]
                dataset = load_neorl_dataset(env, datatype)
            else:
                dataset = qlearning_dataset(env)
            break
        except PermissionError as e:
            print(f'load fail, reloading')
            time.sleep(5)
        except BlockingIOError as e:
            print(f'load fail, reloading')
            time.sleep(5)

    else:
        raise PermissionError(f'load fail!!!')


    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)

    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    critics = []
    for i in range(args.num_q_ensemble):
        critic_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        critics.append(Critic(critic_backbone, args.device))
    critics = torch.nn.ModuleList(critics)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)
    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_reward = None
    dynamics_reward_device = torch.device(args.dynamics_reward_device)
    if args.use_dynamics_reward:
        dynamics_reward = load_discriminator_ensemble(args.task, args.dynamics_reward_path)
        dynamics_reward = DiscriminatorEnsemble(dynamics_reward, ensemble_choosing_interval=args.ensemble_choosing_interval,
                                                max_retain_num=args.max_retain_num, load_min_max=args.load_min_max)
        dynamics_reward.to(device=dynamics_reward_device)
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device,

    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        # penalty_coef=args.penalty_coef,
        dynamics_rewards=dynamics_reward,
        dynamics_reward_factor=args.dynamics_reward_factor,
        reward_use_type=args.reward_use_type,
        resample_num_in_transition=args.resample_num_in_transition,
        d_clip=args.d_clip,
        reward_infer_batch_num=args.reward_infer_batch_num,
        terminal_dynamics_reward=args.terminal_dynamics_reward,
    minimal_rollout_length=args.minimal_rollout_length
    )

    if args.load_dynamics_path:
        if args.load_dynamics_path.startswith('/'):
            dynamics.load(args.load_dynamics_path)
        else:
            dynamics.load(os.path.join(smart_logger.get_base_path(), 'logfile', args.load_dynamics_path))

    # create policy
    policy = MOBILEPolicy(
        dynamics,
        actor,
        critics,
        actor_optim,
        critics_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        penalty_coef=args.penalty_coef,
        num_samples=args.num_samples,
        # TODO: 25 in the paper
        # maybe 40
        state_clip=1e6 if isneorl else 25,
        deterministic_backup=args.deterministic_backup
    )
    actor_backbone_eval = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone_eval = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone_eval = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)

    dist_eval = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor_eval = ActorProb(actor_backbone_eval, dist_eval, 'cpu')
    actor_eval_optim = torch.optim.Adam(actor_eval.parameters(), lr=args.actor_lr)
    critics_eval = []
    for i in range(args.num_q_ensemble):
        critic_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        critics_eval.append(Critic(critic_backbone, args.device))
    critics_eval = torch.nn.ModuleList(critics_eval)
    critics_eval_optim = torch.optim.Adam(critics_eval.parameters(), lr=args.critic_lr)

    policy_eval = MOBILEPolicy(
        None,
        actor_eval,
        critics_eval,
        actor_eval_optim,
        critics_eval_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        penalty_coef=args.penalty_coef,
        num_samples=args.num_samples,
        deterministic_backup=args.deterministic_backup
    )
    # policy_eval = policy

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    fake_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    # log
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    # ----- smart logger init
    origin_info = parameter.information
    now = datetime.now()
    parameter.information = parameter.information + '_MODEL'
    slogger_model = sLogger(log_name=parameter.short_name, log_signature=parameter.signature,
                            logger_category=now.strftime('%Y-%m-%d-%H-%M-%S'),
                            )
    parameter.set_logger(slogger_model)
    parameter.set_config_path(os.path.join(slogger_model.output_dir, 'config'))
    parameter.save_config()
    parameter.information = origin_info
    slogger = sLogger(log_name=parameter.short_name, log_signature=parameter.signature,
                    logger_category=now.strftime('%Y-%m-%d-%H-%M-%S'),
                    )

    parameter.set_logger(slogger)
    parameter.set_config_path(os.path.join(slogger.output_dir, 'config'))
    parameter.save_config()
    slogger.log(parameter)
    # ----
    # create policy trainer
    process_rnd_seed_generator = np.random.RandomState()
    process_rnd_seed_generator.seed(parameters.seed)
    policy_trainer = MBPolicyTrainer(
        policy=policy,
        eval_policy=policy_eval,
        eval_env=env,
        dynamics_eval_env=env_dynamics,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        slogger=slogger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler,
        dynamics_rewards=dynamics_reward,
        process_rnd_generator=process_rnd_seed_generator
    )

    # train
    if not load_dynamics_model:
        dynamics.train(real_buffer.sample_all(), slogger_model, max_epochs_since_update=5, max_epochs=args.dynamics_max_epochs)
    dynamics_model.to(dynamics_reward_device)

    dynamics_model.device = dynamics_reward_device

    policy_trainer.train()


if __name__ == "__main__":

    init_smart_logger()
    parameter = Parameter()
    args = parameter.args
    train(args, parameter)
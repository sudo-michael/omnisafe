import warnings
import torch

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


if __name__ == '__main__':
    experiment_name = 'omnisafe_cartpole'
    eg = ExperimentGrid(exp_name='omnisafe_cartpole')

    # Set the algorithms.
    naive_lagrange_policy = ['PPOLag', 'CPPOPID']
    penalty_policy = ['P3O', 'IPO']


    eg.add('algo', naive_lagrange_policy + penalty_policy)
    eg.add('train_cfgs:device', "cuda")
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [8])
    eg.add('train_cfgs:total_steps', [100_000])
    eg.add('algo_cfgs:steps_per_epoch', [2_000])
    eg.add('logger_cfgs:use_wandb', [True])
    eg.add('logger_cfgs:use_tensorboard', [False])
    eg.add('seed', [0, 1, 2, 3, 4])
    eg.run(train, num_pool=8, gpu_id=[0, 1])

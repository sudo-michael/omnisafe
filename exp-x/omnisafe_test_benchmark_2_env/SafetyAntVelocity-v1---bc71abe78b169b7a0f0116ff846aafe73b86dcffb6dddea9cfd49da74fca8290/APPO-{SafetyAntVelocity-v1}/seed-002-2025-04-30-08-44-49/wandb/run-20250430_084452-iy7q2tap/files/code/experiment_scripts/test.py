import warnings

import torch

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


if __name__ == '__main__':
    experiment_name = 'omnisafe_test_benchmark'
    eg = ExperimentGrid(exp_name='omnisafe_test_benchmark_2_env')

    # Set the algorithms.
    naive_lagrange_policy = ['PPOLag', 'CPPOPID']
    penalty_policy = ['P3O', 'IPO']

    # Set the environments.
    mujoco_envs = [
        'SafetyAntVelocity-v1',
        'SafetyHopperVelocity-v1',
        # 'SafetyHumanoidVelocity-v1',
        # 'SafetyWalker2dVelocity-v1',
        'SafetyHalfCheetahVelocity-v1',
        # 'SafetySwimmerVelocity-v1',
    ]
    eg.add('env_id', mujoco_envs)
    # navigation_envs = [
    #     'SafetyCarGoal0-v0',
    #     'SafetyCarGoal1-v0',
    #     'SafetyCarCircle0-v0',
    #     'SafetyCarCircle1-v0',
    # ]
    # eg.add('env_id', navigation_envs)


    # eg.add('algo', naive_lagrange_policy + penalty_policy)
    eg.add('algo', ['APPO'])
    eg.add('train_cfgs:device', "cuda")
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [8])
    eg.add('train_cfgs:total_steps', [1_000_000])
    eg.add('algo_cfgs:steps_per_epoch', [20_000])
    eg.add('logger_cfgs:use_wandb', [True])
    eg.add('logger_cfgs:use_tensorboard', [False])
    eg.add('seed', [0, 1, 2, 3, 4])
    eg.run(train, num_pool=8, gpu_id=[0, 1])

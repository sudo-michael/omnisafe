# import omnisafe
#
#
# if __name__ == "__main__":
#     env_id = "SafetyPointButton1-v0"
#     custom_cfgs = {
#         "train_cfgs": {
#             "total_steps": 1_000_000,
#             "vector_env_nums": 1,
#             "device": "cuda:0",
#         },
#         "algo_cfgs": {
#             "steps_per_epoch": 2_000,
#             "inner_epoch": 5,
#             "critic_update_iters": 10,
#             "actor_update_iters": 10,
#             "batch_size": 2_000,
#             "eta": 1,
#         },
#         "logger_cfgs": {
#             "use_wandb": False,
#         },
#     }
#     agent = omnisafe.Agent("ALMPG", env_id, custom_cfgs=custom_cfgs)
#     agent.learn()
#     agent.evaluate(num_episodes=5)
#
import warnings

import torch

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


if __name__ == "__main__":
    experiment_name = "omnisafe_test_benchmark_almpg"
    eg = ExperimentGrid(exp_name=experiment_name)

    # Set the environments.
    eg.add("env_id", ["SafetyPointButton1-v0"])
    eg.add("algo", ["ALMPG"])
    eg.add("train_cfgs:device", "cuda")
    eg.add("train_cfgs:vector_env_nums", [1])
    eg.add("train_cfgs:torch_threads", [8])
    eg.add("train_cfgs:total_steps", [1_000_000])
    eg.add("algo_cfgs:steps_per_epoch", [2_000])
    eg.add("algo_cfgs:inner_epoch", [5, 10])
    eg.add("algo_cfgs:critic_update_iters", [5, 10])
    eg.add("algo_cfgs:actor_update_iters", [5, 10])
    eg.add("algo_cfgs:batch_size", [2_000])
    eg.add("algo_cfgs:eta", [0.1, 0.5, 1.0])
    eg.add("logger_cfgs:use_wandb", [True])
    eg.add("logger_cfgs:use_tensorboard", [False])
    eg.add("seed", [0, 1, 2])
    eg.run(train, num_pool=8, gpu_id=[0, 1])

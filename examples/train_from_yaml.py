# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of training a policy from default config yaml with OmniSafe."""

import omnisafe


if __name__ == "__main__":
    env_id = "SafetyAntVelocity-v1"
    custom_cfgs = {
        "train_cfgs": {
            "total_steps": 1_000_000,
            "vector_env_nums": 4,
        },
        "algo_cfgs": {
            "steps_per_epoch": 20_000,
            "update_iters": 1,
        },
        "logger_cfgs": {
            "use_wandb": False,
        },
    }
    agent = omnisafe.Agent("APPO", env_id)
    agent.learn()

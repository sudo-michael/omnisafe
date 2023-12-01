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
"""OffPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from omnisafe.envs.core import CMDP, make, support_envs
from omnisafe.envs.wrapper import (
    ActionScale,
    AutoReset,
    CostNormalize,
    ObsNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config
from omnisafe.utils.tools import get_device



from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config


class ShieldOnlineAdapter:
    """Filtered Online Adapter for OmniSafe.

    OmniSafe is a framework for safe reinforcement learning. It is designed to be compatible with
    any existing RL algorithms. The online adapter is used to adapt the environment to the
    framework.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of parallel environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnlineAdapter`."""
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._cfgs: Config = cfgs

        self._device: torch.device = get_device(cfgs.train_cfgs.device)
        self._env_id: str = env_id
        self._env: CMDP = make(env_id, num_envs=num_envs, device=self._device, **self._cfgs['env_cfgs']) # tell cmdp env to use shield wrapper
        self._eval_env: CMDP = make(env_id, num_envs=1, device=self._device, **self._cfgs['env_cfgs'])

        self._wrapper(
            obs_normalize=cfgs.algo_cfgs.obs_normalize,
            reward_normalize=cfgs.algo_cfgs.reward_normalize,
            cost_normalize=cfgs.algo_cfgs.cost_normalize,
        )

        self._env.set_seed(seed)

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        """Wrapper the environment.

        .. hint::
            OmniSafe supports the following wrappers:

        +-----------------+--------------------------------------------------------+
        | Wrapper         | Description                                            |
        +=================+========================================================+
        | TimeLimit       | Limit the time steps of the environment.               |
        +-----------------+--------------------------------------------------------+
        | AutoReset       | Reset the environment when the episode is done.        |
        +-----------------+--------------------------------------------------------+
        | ObsNormalize    | Normalize the observation.                             |
        +-----------------+--------------------------------------------------------+
        | RewardNormalize | Normalize the reward.                                  |
        +-----------------+--------------------------------------------------------+
        | CostNormalize   | Normalize the cost.                                    |
        +-----------------+--------------------------------------------------------+
        | ActionScale     | Scale the action.                                      |
        +-----------------+--------------------------------------------------------+
        | Unsqueeze       | Unsqueeze the step result for single environment case. |
        +-----------------+--------------------------------------------------------+


        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to True.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, time_limit=1000, device=self._device)
            self._eval_env = TimeLimit(self._eval_env, time_limit=1000, device=self._device)
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
            self._eval_env = AutoReset(self._eval_env, device=self._device)
        if obs_normalize:
            self._env = ObsNormalize(self._env, device=self._device)
            self._eval_env = ObsNormalize(self._eval_env, device=self._device)
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        self._env = ActionScale(self._env, low=-1.0, high=1.0, device=self._device)
        self._eval_env = ActionScale(self._eval_env, low=-1.0, high=1.0, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)
        self._eval_env = Unsqueeze(self._eval_env, device=self._device)

    @property
    def action_space(self) -> OmnisafeSpace:
        """The action space of the environment."""
        return self._env.action_space

    @property
    def observation_space(self) -> OmnisafeSpace:
        """The observation space of the environment."""
        return self._env.observation_space

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        return self._env.step(action)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        return self._env.reset(seed=seed, options=options)

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the important components of the environment.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be an empty dict. common wrappers are
            ``obs_normalize``, ``reward_normalize``, and ``cost_normalize``.

        Returns:
            The saved components of environment, e.g., ``obs_normalizer``.
        """
        return self._env.save()





class ShieldOffPolicyAdapter(ShieldOnlineAdapter):
    """OffPolicy Adapter for OmniSafe.

    :class:`OffPolicyAdapter` is used to adapt the environment to the off-policy training.

    .. note::
        Off-policy training need to update the policy before finish the episode,
        so the :class:`OffPolicyAdapter` will store the current observation in ``_current_obs``.
        After update the policy, the agent will *remember* the current observation and
        use it to interact with the environment.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _current_obs: torch.Tensor
    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor
    _total_cost: int = 0

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize a instance of :class:`OffPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._current_obs, _ = self.reset()
        self._max_ep_len: int = 1000
        self._reset_log()

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic,
        logger: Logger,
    ) -> None:
        """Rollout the environment with deterministic agent action.

        Args:
            episode (int): Number of episodes.
            agent (ConstraintActorCritic): Agent.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        for _ in range(episode):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
            obs, _ = self._eval_env.reset()
            obs = obs.to(self._device)

            done = False
            while not done:
                act = agent.step(obs, deterministic=True)
                obs, reward, cost, terminated, truncated, info = self._eval_env.step(act)
                obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (obs, reward, cost, terminated, truncated)
                )
                ep_ret += info.get('original_reward', reward).cpu()
                ep_cost += info.get('original_cost', cost).cpu()
                ep_len += 1
                done = bool(terminated[0].item()) or bool(truncated[0].item())

            
            logger.store(
                {
                    'Metrics/TestEpRet': ep_ret,
                    'Metrics/TestEpCost': ep_cost,
                    'Metrics/TestEpLen': ep_len,
                },
            )

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            rollout_step (int): Number of rollout steps.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor, reward critic,
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            use_rand_action (bool): Whether to use random action.
        """
        for _ in range(rollout_step):
            if use_rand_action:
                act = torch.as_tensor(self._env.sample_action(), dtype=torch.float32).to(
                    self._device,
                )
            else:
                act = agent.step(self._current_obs, deterministic=False)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)
            real_next_obs = next_obs.clone()
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    if 'final_observation' in info:
                        real_next_obs[idx] = info['final_observation'][idx]
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)

            buffer.store(
                obs=self._current_obs,
                act=act,
                reward=reward,
                cost=cost,
                done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                next_obs=real_next_obs,
            )

            self._current_obs = next_obs

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1

        self._total_cost += info.get('original_cost', cost).cpu()


    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
                'Metrics/TotalCost': self._total_cost,
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
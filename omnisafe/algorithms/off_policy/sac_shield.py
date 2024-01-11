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
"""Implementation of the Lagrangian version of Soft Actor-Critic algorithm."""


import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac_lag import SACLag
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.adapter.shield_offpolicy_adapter import ShieldOffPolicyAdapter

@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class SACShield(SACLag):
    def _init(self) -> None:
        super()._init()
        self._actor_critic = ConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

        self._env = ShieldOffPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
    def _init_log(self) -> None:
        super()._init_log()
        # NOTE: added for atu, min_and_max since logger prints running average, which is useless for a total
        # self._logger.register_key('Atu/TotalCollidePersuer', window_length=1)
        # self._logger.register_key('Atu/TotalUseHJ', window_length=1)
        # self._logger.register_key('Atu/TotalReachGoal', window_length=1) 
    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. note::

            +----------+---------------------------------------+
            | obs      | ``observaion`` stored in buffer.      |
            +==========+=======================================+
            | act      | ``action`` stored in buffer.          |
            +----------+---------------------------------------+
            | reward   | ``reward`` stored in buffer.          |
            +----------+---------------------------------------+
            | cost     | ``cost`` stored in buffer.            |
            +----------+---------------------------------------+
            | next_obs | ``next observaion`` stored in buffer. |
            +----------+---------------------------------------+
            | done     | ``terminated`` stored in buffer.      |
            +----------+---------------------------------------+

        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the ``update_iters`` times.
        """
        for ii in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)

        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.update_lagrange_multiplier(Jc)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update cost critic.

        - Get the TD loss of cost critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            cost (torch.Tensor): The ``cost`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q1_value_c, next_q2_value_c = self._actor_critic.target_cost_critic(next_obs, next_action)
            target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * torch.min(next_q1_value_c, next_q2_value_c)
        q1_value_c, q2_value_c = self._actor_critic.cost_critic(obs, action)
        loss = nn.functional.mse_loss(q1_value_c, target_q_value_c) + nn.functional.mse_loss(q2_value_c, target_q_value_c)


        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q1_value_c.mean().item(),
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in SACLag is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \lambda Q^C (s, \pi (s))

        where :math:`Q^V` is the min value of two reward critic networks outputs, :math:`Q^C` is the
        value of cost critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        loss_q_r_1, loss_q_r_2 = self._actor_critic.reward_critic(obs, action)
        loss_r = self._alpha * log_prob - torch.min(loss_q_r_1, loss_q_r_2) 
        loss_q_c = torch.min(*self._actor_critic.cost_critic(obs, action))
        loss_c = self._lagrange.lagrangian_multiplier.item() * loss_q_c 

        return (loss_r + loss_c).mean() / (1 + self._lagrange.lagrangian_multiplier.item())
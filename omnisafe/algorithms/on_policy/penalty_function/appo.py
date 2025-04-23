"""Implementation of the APPO algorithm."""

import numpy as np
import torch
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed


@registry.register
class APPO(PolicyGradient):
    """The Implementation of the APPO algorithm.

    References:
        - Title: Augmented Proximal Policy Optimization for Safe Reinforcement Learning
        - Authors: Juntao Dai, Jiaming Ji, Long Yang, Qian Zheng, Gang Pan
        - URL: `APPO <https://ojs.aaai.org/index.php/AAAI/article/view/25888>`_
        - Authors Implementation: https://github.com/calico-1226/appo/tree/main
    """

    def _init(self) -> None:
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        self._tau: float = self._cfgs.augmented_lagrangian_cfgs.tau

    def _init_log(self) -> None:
        """Log the APPO specific information.

        +-------------------+-----------------------------------+
        | Things to log     | Description                       |
        +===================+===================================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key("Metrics/LagrangeMultiplier", min_and_max=True)

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. hint::

            +----------------+------------------------------------------------------------------+
            | obs            | ``observation`` sampled from buffer.                             |
            +================+==================================================================+
            | act            | ``action`` sampled from buffer.                                  |
            +----------------+------------------------------------------------------------------+
            | target_value_r | ``target reward value`` sampled from buffer.                     |
            +----------------+------------------------------------------------------------------+
            | target_value_c | ``target cost value`` sampled from buffer.                       |
            +----------------+------------------------------------------------------------------+
            | logp           | ``log probability`` sampled from buffer.                         |
            +----------------+------------------------------------------------------------------+
            | adv_r          | ``estimated advantage`` (e.g. **GAE**) sampled from buffer.      |
            +----------------+------------------------------------------------------------------+
            | adv_c          | ``estimated cost advantage`` (e.g. **GAE**) sampled from buffer. |
            +----------------+------------------------------------------------------------------+


        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the data from buffer.
        #. Shuffle the data and split it into mini-batch data.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.
        #. Repeat steps 2, 3, 4 until the KL divergence violates the limit.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats("Metrics/EpCost")[0]
        assert not np.isnan(Jc), "cost for updating lagrange multiplier is nan"
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)
        # then update the policy and value function
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data["obs"],
            data["act"],
            data["logp"],
            data["target_value_r"],
            data["target_value_c"],
            data["adv_r"],
            data["adv_c"],
        )

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description="Updating..."):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                # https://github.com/calico-1226/appo/blob/main/appo/algorithms/policy_gradient.py#L512
                self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f"Early stopping at iter {i + 1} due to reaching max kl")
                break

        self._logger.store(
            {
                "Train/StopIter": update_counts,  # pylint: disable=undefined-loop-variable
                "Value/Adv": adv_r.mean().item(),
                "Train/KL": final_kl,
            },
        )
        for _ in track(range(self._cfgs.algo_cfgs.update_iters_value), description="Updating..."):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                # https://github.com/calico-1226/appo/blob/main/appo/algorithms/appo.py#L90
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    # https://github.com/calico-1226/appo/blob/main/appo/algorithms/appo.py#L91
                    self._update_cost_critic(obs, target_value_c)

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        #. Compute the loss function.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
            the ``KL divergence`` between the old policy and the new policy is calculated.
            And the ``KL divergence`` is used to determine whether the update is successful.
            If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        ratio_cliped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )

        penalty = self._lagrange.lagrangian_multiplier.item()
        Jc = self._logger.get_stats("Metrics/EpCost")[0]
        if Jc - self._lagrange.cost_limit + penalty / self._tau > 0:
            alm_penalty = penalty + self._tau * (Jc - self._lagrange.cost_limit)
        else:
            alm_penalty = 0.0

        loss_adv_r = torch.min(ratio * adv_r, ratio_cliped * adv_r).mean()
        loss_adv_c = torch.max(ratio * adv_c, ratio_cliped * adv_c).mean()
        loss_alm = alm_penalty * loss_adv_c
        loss = (-loss_adv_r + loss_adv_c) / (1 + penalty)
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()
        # useful extra info
        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                "Train/Entropy": entropy,
                "Train/PolicyRatio": ratio,
                "Train/PolicyStd": std,
                "Loss/Loss_pi": loss.mean().item(),
            },
        )

        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

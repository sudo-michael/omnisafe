from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rich.progress import track
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed


def get_grad_list(params, centralize_grad=False):
    grad_list = []
    for p in params:
        g = p.grad
        if g is None:
            g = 0.0
        else:
            g = p.grad.data
            if len(list(g.size())) > 1 and centralize_grad:
                # centralize grads
                g.add_(-g.mean(dim=tuple(range(1, len(list(g.size())))), keepdim=True))
        grad_list += [g]
    return grad_list


def compute_grad_norm(grad_list, centralize_grad_norm=False):
    grad_norm = 0.0
    for g in grad_list:
        if g is None or (isinstance(g, float) and g == 0.0):
            continue

        if g.dim() > 1 and centralize_grad_norm:
            # centralize grads
            g.add_(-g.mean(dim=tuple(range(1, g.dim())), keepdim=True))

        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def armijo_line_search(f, model, alpha_max, c, beta):
    init_net = deepcopy(model)
    init_loss = f(init_net)
    init_loss.backward()

    init_params = list(init_net.parameters())
    init_grad_list = get_grad_list(init_params)
    init_grad_norm = compute_grad_norm(init_grad_list)

    # applying in-place updates to the parameters of `model`
    net = model
    armijo_condition_satisfied = False
    alpha = alpha_max

    iters = 0
    with torch.no_grad():
        while not armijo_condition_satisfied:
            # compute prospective parameters
            for i, (p_next, p_current) in enumerate(
                zip(net.parameters(), init_net.parameters())
            ):
                p_next.data = p_current - alpha * init_grad_list[i]

            next_loss = f(net)
            lhs = next_loss
            rhs = init_loss - c * alpha * init_grad_norm**2
            if lhs <= rhs:
                armijo_condition_satisfied = True
            else:
                alpha = alpha * beta
                iters += 1
                if iters > 500:
                    print("iters > 500 breaking")
                    break
    return next_loss


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods,line-too-long
class SPMA(BaseAlgo):
    """SPMA"""

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OnPolicyAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: OnPolicyAdapter = OnPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, (
            "The number of steps per epoch is not divisible by the number of environments."
        )
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        self._actor_critic: ConstraintActorCritic = ConstraintActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init(self) -> None:
        """The initialization of the algorithm."""
        self._buf: VectorOnPolicyBuffer = VectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

        self._alpha_max = self._cfgs.algo_cfgs.armijo_alpha_max

    def _init_log(self) -> None:
        """Log info about epoch.
        +-----------------------+----------------------------------------------------------------------+
        | Things to log         | Description                                                          |
        +=======================+======================================================================+
        | Train/Epoch           | Current epoch.                                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpCost        | Average cost of the epoch.                                           |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpRet         | Average return of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpLen         | Average length of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Values/reward         | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-----------------------+----------------------------------------------------------------------+
        | Values/cost           | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-----------------------+----------------------------------------------------------------------+
        | Values/Adv            | Average reward advantage of the epoch.                               |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi          | Loss of the policy network.                                          |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic | Loss of the cost critic network.                                     |
        +-----------------------+----------------------------------------------------------------------+
        | Train/Entropy         | Entropy of the policy network.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Train/PolicyRatio     | Ratio of the policy network.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Train/LR              | Learning rate of the policy network.                                 |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/Seed             | Seed of the experiment.                                              |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps    | Total steps of the experiment.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Time                  | Total time.                                                          |
        +-----------------------+----------------------------------------------------------------------+
        | FPS                   | Frames per second of the epoch.                                      |
        +-----------------------+----------------------------------------------------------------------+
        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save["pi"] = self._actor_critic.actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()["obs_normalizer"]
            what_to_save["obs_normalizer"] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key(
            "Metrics/EpRet",
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            "Metrics/EpCost",
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            "Metrics/EpLen",
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

        self._logger.register_key("Train/Epoch")
        self._logger.register_key("Train/Entropy")
        self._logger.register_key("Train/KL")
        self._logger.register_key("Train/PolicyRatio", min_and_max=True)
        self._logger.register_key("Train/LR")
        if self._cfgs.model_cfgs.actor_type == "gaussian_learning":
            self._logger.register_key("Train/PolicyStd")

        self._logger.register_key("TotalEnvSteps")

        # log information about actor
        self._logger.register_key("Loss/Loss_pi", delta=True)
        self._logger.register_key("Value/Adv")

        # log information about critic
        self._logger.register_key("Loss/Loss_reward_critic", delta=True)
        self._logger.register_key("Value/reward")

        self._logger.register_key("Time/Total")
        self._logger.register_key("Time/Rollout")
        self._logger.register_key("Time/Update")
        self._logger.register_key("Time/Epoch")
        self._logger.register_key("Time/FPS")

        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self.logger.register_key(env_spec_key)

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log("INFO: Start training")

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()
            for inner_epoch in range(self._cfgs.algo_cfgs.inner_epoch):
                rollout_time = time.time()
                self._env.rollout(
                    steps_per_epoch=self._steps_per_epoch,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                )
                self._logger.store({"Time/Rollout": time.time() - rollout_time})

                update_time = time.time()
                self._update()
                self._logger.store({"Time/Update": time.time() - update_time})

            # increase dual variable
            # Jc = discount_cumsum(self._buf.data["cost"], self._cfgs.algo_cfgs.gamma)
            Jc = self._logger.get_stats("Metrics/EpCost")[0]
            assert not np.isnan(Jc), "Jc is nan"
            self._lagrangian_multiplier = max(
                self._lagrangian_multiplier + self._tau * (Jc - self._cost_limit), 0
            )

            # increase augmented Lagrangian regualrizer
            self._tau = np.clip(
                self._cfgs.augmented_lagrange_cfgs.beta * max(0, Jc - self._cost_limit),
                self._cfgs.augmented_lagrange_cfgs.tau_init,
                self._cfgs.augmented_lagrange_cfgs.tau_max,
            )

            self._logger.store(
                {
                    "TotalEnvSteps": (epoch + 1)
                    * self._cfgs.algo_cfgs.steps_per_epoch
                    * self._cfgs.algo_cfgs.inner_epoch,
                    "Time/FPS": self._cfgs.algo_cfgs.steps_per_epoch
                    / (time.time() - epoch_time),
                    "Time/Total": (time.time() - start_time),
                    "Time/Epoch": (time.time() - epoch_time),
                    "Train/Epoch": epoch,
                    "Train/LR": (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                    "Metrics/AugmentedLagrangianRegularizer": self._tau,
                    "Metrics/LagrangeMultiplier": self._lagrangian_multiplier,
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats("Metrics/EpRet")[0]
        ep_cost = self._logger.get_stats("Metrics/EpCost")[0]
        ep_len = self._logger.get_stats("Metrics/EpLen")[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

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
        """
        data = self._buf.get()
        obs, act, logp, target_value_r, adv_r = (
            data["obs"],
            data["act"],
            data["logp"],
            data["target_value_r"],
            data["adv_r"],
        )

        original_obs = obs
        with torch.no_grad():
            old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, adv_r),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for i in track(
            range(self._cfgs.algo_cfgs.critic_update_iters),
            description="Updating critics...",
            disable=True
        ):
            for (
                obs,
                _,
                _,
                target_value_r,
                _,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)

        for i in track(
            range(self._cfgs.algo_cfgs.actor_update_iters),
            description="Updating actor...",
            disable=True
        ):
            for (
                obs,
                act,
                log_pi_t,
                _,
                adv_r,
            ) in dataloader:
                self._update_actor(obs, act, log_pi_t, adv_r)

        new_distribution = self._actor_critic.actor(original_obs)

        kl = (
            torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
            .sum(-1, keepdim=True)
            .mean()
        )
        kl = distributed.dist_avg(kl)

        final_kl = kl.item()

        self._logger.store(
            {
                "Value/Adv": adv_r.mean().item(),
                "Train/KL": final_kl,
            },
        )

    def _update_reward_critic(
        self, obs: torch.Tensor, target_value_r: torch.Tensor
    ) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_r (torch.Tensor): The ``target_value_r`` sampled from buffer.
        """
        if not self._cfgs.algo_cfgs.use_armijo_critic:
            self._actor_critic.reward_critic_optimizer.zero_grad()
            loss = nn.functional.mse_loss(
                self._actor_critic.reward_critic(obs)[0], target_value_r
            )
            loss.backward()

            distributed.avg_grads(self._actor_critic.reward_critic)
            self._actor_critic.reward_critic_optimizer.step()
            self._logger.store({"Loss/Loss_reward_critic": loss.mean().item()})
            return

        def f(model):
            return nn.functional.mse_loss(model(obs)[0], target_value_r)

        loss = armijo_line_search(
            f,
            self._actor_critic.reward_critic,
            self._alpha_max,
            self._cfgs.algo_cfgs.armijo_c,
            self._cfgs.algo_cfgs.armijo_beta,
        )
        self._logger.store({"Loss/Loss_reward_critic": loss.mean().item()})

    def _update_cost_critic(
        self, obs: torch.Tensor, target_value_c: torch.Tensor
    ) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_c (torch.Tensor): The ``target_value_c`` sampled from buffer.
        """
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(
            self._actor_critic.cost_critic(obs)[0], target_value_c
        )
        loss.backward()

        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({"Loss/Loss_cost_critic": loss.mean().item()})

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_pi_t: torch.Tensor,
        adv_r: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
        """
        adv = adv_r

        if not self._cfgs.algo_cfgs.use_armijo_actor:
            distribution = self._actor_critic.actor(obs)
            log_pitheta = self._actor_critic.actor.log_prob(act)
            std = self._actor_critic.actor.std
            ratio = torch.exp(log_pitheta - log_pi_t)  # \pi_{\theta} / \pi_t

            linear_loss = ratio * adv
            eta = self._cfgs.algo_cfgs.eta
            # KL approx http://joschu.net/blog/kl-approx.html
            # KL(pi_t || pi_theta) ~= (r - 1) - log r, where r = pi_theta / pi_t
            bregman_divergence_loss = 1 / eta * (ratio - 1 - (log_pitheta - log_pi_t))

            loss = -(linear_loss + bregman_divergence_loss).mean()
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
            distributed.avg_grads(self._actor_critic.actor)
            self._actor_critic.actor_optimizer.step()
            return

        eta = self._cfgs.algo_cfgs.eta
        def f(model):
            distribution = model(obs)
            log_pitheta = model.log_prob(act)
            ratio = torch.exp(log_pitheta - log_pi_t)  # \pi_{\theta} / \pi_t

            linear_loss = ratio * adv
            bregman_divergence_loss = 1 / eta * (ratio - 1 - (log_pitheta - log_pi_t))
            loss = -(linear_loss + bregman_divergence_loss).mean()
            return loss

        # print current weights and output
        for p in self._actor_critic.actor.parameters():
            print(p)
        with torch.no_grad():
            print(self._actor_critic.actor(obs))
        loss = armijo_line_search(
            f,
            self._actor_critic.actor,
            self._alpha_max,
            self._cfgs.algo_cfgs.armijo_c,
            self._cfgs.algo_cfgs.armijo_beta,
        )
        self._logger.store({"Loss/Loss_pi": loss.mean().item()})
